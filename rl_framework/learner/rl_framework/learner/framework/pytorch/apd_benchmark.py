# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import os

from rl_framework.learner.dataset.network_dataset.pytorch.network_dataset_random import (
    NetworkDataset,
)
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.learner.framework.common.log_manager import LogManager
from rl_framework.learner.framework.pytorch.apd_datasets import DataPrefetcher, Datasets
from rl_framework.learner.framework.pytorch.model_manager import ModelManager
from rl_framework.learner.framework.pytorch.step_context import StepContext
from rl_framework.common.logging import logger as LOG


class Profiler:
    def __init__(self, active, rank) -> None:
        self.active = active
        self.step_count = 0
        if self.active:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=10, warmup=2, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./log/torch_profile", "worker_%s" % rank
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            self.profiler.start()

    def step(self):
        if self.active:
            self.step_count += 1
            self.profiler.step()
            if self.step_count >= 50:
                self.profiler.stop()
                LOG.info("profile file dump finished")
                self.active = False


class Benchmark(object):
    def __init__(
        self,
        network,
        dataset,
        log_manager: LogManager,
        model_manager: ModelManager,
        config_manager: ConfigControl,
        node_info=None,
        slow_time: float = 0.0,
    ):
        self.log_manager = log_manager
        LOG.info("init starting, backend=pytorch")
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.slow_time = slow_time

        self.dataset_base = dataset
        self.init_env(node_info)
        self._init_model(network)
        if torch.cuda.is_available():
            self.dataset = DataPrefetcher(
                self.dataset_base,
                self.device,
                self.config_manager.use_fp16,
            )
        else:
            self.dataset = Datasets(self.dataset_base)
        self.step_train_times = list()
        self.skip_update_times = 0
        self._last_save_model_time = 0
        LOG.info("init finished")

    def init_env(self, node_info):
        self.node = node_info
        self.distributed_backend = self.config_manager.distributed_backend
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.node.local_rank)
            torch.cuda.set_device(self.node.local_rank)
        else:
            self.device = torch.device("cpu", self.node.local_rank)
        LOG.info(f"Use {self.device} as default device")
        if self.config_manager.use_mix_precision:
            LOG.info("Use auto mix precision")

    def _default_optimizer(self):
        initial_lr = self.net.learning_rate
        parameters = self.net.parameters()
        optimizer = torch.optim.Adam(
            params=parameters, lr=initial_lr, betas=(0.9, 0.999), eps=1e-8
        )
        return optimizer

    def _init_model(self, network):
        self.local_step = 0
        self.net = network.to(self.device)
        if self.config_manager.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)
        if self.config_manager.use_compile and hasattr(torch, "compile"):
            self.net = torch.compile(self.net)
        get_optimizer = getattr(self.net, "get_optimizer", None)
        if callable(get_optimizer):
            self.optimizer = self.net.get_optimizer()
        else:
            self.optimizer = self._default_optimizer()
        self.parameters = [
            p
            for param_group in self.optimizer.param_groups
            for p in param_group["params"]
        ]
        # load init model
        if self.config_manager.use_init_model:
            model_checkpoint_path = os.path.join(
                self.config_manager.init_model_path, "model.pth"
            )
            ckpt_step = self.model_manager.restore_model_and_optimizer(
                self.net, self.optimizer, model_checkpoint_path
            )
            self.local_step = 0 if ckpt_step is None else ckpt_step
        self.init_step = self.local_step
        get_lr_scheduler = getattr(self.net, "get_lr_scheduler", None)
        if callable(get_lr_scheduler):
            self.lr_scheduler = self.net.get_lr_scheduler(
                self.optimizer, self.local_step
            )
        else:
            self.lr_scheduler = None

        if self.distributed_backend == "horovod" and self.node.has_hvd:
            self.net.to(memory_format=torch.contiguous_format)
            import horovod.torch as hvd

            self.optimizer = hvd.DistributedOptimizer(self.optimizer)
            hvd.broadcast_parameters(self.net.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            self.net_wrapper = self.net
        elif self.distributed_backend == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.net_wrapper = DDP(
                self.net,
                [
                    self.node.local_rank,
                ]
                if torch.cuda.is_available()
                else None,
                self.node.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.config_manager.has_unused_params,
            )
        else:
            self.net_wrapper = self.net
        self.local_step = torch.tensor(self.local_step, dtype=torch.int).to(self.device)
        if self.config_manager.use_jit:
            example_data = torch.from_numpy(
                NetworkDataset(
                    self.config_manager, self.dataset_base.adapter
                ).get_next_batch()
            ).to(self.device)
            example_data_list = self.net.format_data(example_data)
            if self.config_manager.use_mix_precision:
                with torch.cuda.amp.autocast(cache_enabled=False):
                    torch._C._jit_set_autocast_mode(False)
                    self.net_wrapper = torch.jit.trace(
                        self.net_wrapper, (example_data_list)
                    )
            else:
                self.net_wrapper = torch.jit.trace(
                    self.net_wrapper, (example_data_list)
                )
        if self.config_manager.print_variables and self.node.is_chief_rank:
            self.model_manager.print_variables(
                self.net_wrapper, self.optimizer, self.local_step
            )
        # only save checkpoint_0 on master node
        if self.node.is_chief_rank:
            self.model_manager.save_checkpoint(self.net, self.optimizer, 0)
            self._last_save_model_time = time.time()

    def do_train_step(self, step_context: StepContext, _input_datas):
        self.optimizer.zero_grad()
        results = {}
        if self.config_manager.use_mix_precision:
            with torch.cuda.amp.autocast():
                data_list = self.net.format_data(_input_datas)
                rst_list = self.net_wrapper(data_list)
                total_loss, info_list = self.net.compute_loss(data_list, rst_list)
                results["total_loss"] = total_loss.item()
            self.scaler.scale(total_loss).backward()
            if (
                self.distributed_backend == "horovod" and self.node.has_hvd
            ):  # only horovod mode needs
                self.optimizer.synchronize()
            if self.config_manager.use_grad_clip:  # grad clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.parameters, self.config_manager.grad_clip_range
                )
            if self.local_step % self.config_manager.display_every == 0:
                step_context.set_forward_info(total_loss, info_list)
                if self.config_manager.check_values:
                    step_context.check_has_inf_nan(total_loss, self.parameters)
            if self.distributed_backend == "horovod" and self.node.has_hvd:
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
            else:
                self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            data_list = self.net.format_data(_input_datas)
            rst_list = self.net_wrapper(data_list)
            total_loss, info_list = self.net.compute_loss(data_list, rst_list)
            results["total_loss"] = total_loss.item()
            total_loss.backward()
            if (
                self.distributed_backend == "horovod" and self.node.has_hvd
            ):  # only horovod mode needs
                self.optimizer.synchronize()
            if self.config_manager.use_grad_clip:  # grad clip
                torch.nn.utils.clip_grad_norm_(
                    self.parameters, self.config_manager.grad_clip_range
                )
            if self.local_step % self.config_manager.display_every == 0:
                step_context.set_forward_info(total_loss, info_list)
                if self.config_manager.check_values:
                    step_context.check_has_inf_nan(total_loss, self.parameters)
            if self.distributed_backend == "horovod" and self.node.has_hvd:
                with self.optimizer.skip_synchronize():
                    self.optimizer.step()
            else:
                self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)
        results["info_list"] = _info_list
        return results

    def _check_save_model(self):
        return (
            self.local_step % self.config_manager.save_model_steps == 0
            or time.time() - self._last_save_model_time
            > self.config_manager.save_model_seconds
        )

    def _do_train(self):
        LOG.info("Start training...")
        self.net_wrapper.train()
        start_time = time.time()
        profiler = Profiler(self.config_manager.dump_profile, self.node.rank)
        step_context = StepContext(
            self.node.rank,
            self.node.local_rank,
            self.node.rank_size,
            self.config_manager.ips[0],
            self.config_manager.batch_size,
        )
        _input_datas = self.dataset.next()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.config_manager.use_mix_precision
        )

        first_step = True
        for _ in range(self.config_manager.warmup_steps, self.config_manager.max_steps):
            batch_begin = time.time()
            if self.slow_time > 0:
                time.sleep(self.slow_time)
            results = self.do_train_step(step_context, _input_datas)
            _input_datas = self.dataset.next()
            batch_duration = time.time() - batch_begin
            profiler.step()
            self.local_step += 1

            if first_step:
                first_step = False
            else:
                self.step_train_times.append(batch_duration)

            if self.node.is_chief_rank and (
                self.local_step == 0
                or self.local_step % self.config_manager.display_every == 0
            ):
                results["ip"] = self.config_manager.ips[0]
                results["batch_size"] = self.config_manager.batch_size
                results["step"] = self.local_step
                results["gpu_nums"] = self.node.rank_size
                results["sample_recv_speed"] = self.dataset.get_recv_speed()
                results["sample_consume_speed"] = self.get_sample_consume_speed(
                    self.config_manager.batch_size, self.step_train_times
                )
                self.log_manager.print_result(results)

            if self._check_save_model() and self.node.is_chief_rank:
                self.model_manager.save_checkpoint(
                    self.net, self.optimizer, self.local_step
                )
                self._last_save_model_time = time.time()

        # training finished
        images_per_sec = (
            (time.time() - start_time)
            / (self.config_manager.max_steps - self.config_manager.warmup_steps)
            * self.config_manager.batch_size
        )
        LOG.info("-" * 64)
        LOG.info("total images/sec: %.2f" % images_per_sec)
        LOG.info("-" * 64)
        # Save the model checkpoint.
        if self.node.is_chief_rank:
            self.model_manager.save_checkpoint(
                self.net, self.optimizer, self.local_step
            )
            self._last_save_model_time = time.time()

    def run(self):
        self._do_train()

    def get_sample_consume_speed(self, batch_size, step_train_times, scale=1):
        if not step_train_times:
            return 0
        if len(step_train_times) <= 1:
            return step_train_times[0]
        times = np.array(step_train_times[1:])
        speed_mean = scale * batch_size / np.mean(times)
        return speed_mean
