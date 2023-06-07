# -*- coding: utf-8 -*-
import time

try:
    import horovod.torch as hvd

    has_hvd = True
except:
    has_hvd = False
import numpy as np
import torch
import os

from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.learner.framework.common.log_manager import LogManager
from rl_framework.learner.framework.pytorch.apd_datasets import Datasets
from rl_framework.learner.framework.pytorch.model_manager import ModelManager
from rl_framework.learner.framework.pytorch.node_info import NodeInfo



class Benchmark(object):
    def __init__(
        self,
        network,
        dataset,
        log_manager: LogManager,
        model_manager: ModelManager,
        config_manager: ConfigControl,
        node_info: NodeInfo,
    ):
        if has_hvd:
            config_manager.reset_hvd_rank(
                hvd.rank(), hvd.size(), hvd.local_rank(), hvd.local_size()
            )

        self.log_manager = log_manager
        self.log_manager.print_info("init starting, backend=pytorch")
        self.config_manager = config_manager
        self.model_manager = model_manager

        self.rank = node_info.rank
        self.rank_size = node_info.size
        self.local_rank = node_info.local_rank
        self.local_size = node_info.local_size
        self.is_chief_rank = self.rank == 0

        device_idx = self.local_rank
        if torch.cuda.is_available():
            self.log_manager.print_info("Cuda is available and being used")
            self.device = torch.device("cuda", device_idx)
            torch.cuda.set_device(device_idx)
        else:
            self.log_manager.print_info("Cuda not available. Using CPU instead")
            self.device = torch.device("cpu", device_idx)

        self.net = network.to(self.device)
        self.dataset = Datasets(dataset)
        self.optimizer = self._init_optimizer()
        self._init_model()

        self.local_step = 0
        self.step_train_times = list()
        self.skip_update_times = 0
        self.log_manager.print_info("init finished")

    def _init_optimizer(self):
        initial_lr = self.net.learning_rate
        parameters = self.net.parameters()
        optimizer = torch.optim.Adam(
            params=parameters, lr=initial_lr, betas=(0.9, 0.999), eps=1e-8
        )
        if has_hvd:
            optimizer = hvd.DistributedOptimizer(optimizer, self.net.named_parameters())
        return optimizer

    def _init_model(self):

        # only load checkpoint on master node and then broadcast
        if self.config_manager.use_init_model:
            model_checkpoint_path = os.path.join(
                self.config_manager.init_model_path, "model.pth"
            )
            self.log_manager.print_info(
                f"Loading checkpoint from {model_checkpoint_path}"
            )
            self.load_checkpoint(model_checkpoint_path)

        if has_hvd:
            hvd.broadcast_parameters(self.net.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        # only save checkpoint_0 on master node
        self.log_manager.print_info(
            f"Saving checkpoint_0 to {self.config_manager.save_model_dir}"
        )
        os.makedirs(self.config_manager.save_model_dir, exist_ok=True)
        self.save_checkpoint(self.config_manager.save_model_dir)
        self.model_manager.send_model(
            self.config_manager.save_model_dir, self.config_manager.send_model_dir
        )

    def _do_train(self):
        self.log_manager.print_info("Start training...")
        self.net.train()
        start_time = time.time()
        for _ in range(self.config_manager.warmup_steps, self.config_manager.max_steps):
            batch_begin = time.time()
            self.optimizer.zero_grad()
            results = {}
            input_datas = self.dataset.next_batch()
            _input_datas = torch.from_numpy(input_datas).to(
                dtype=torch.float32, device=self.device
            )
            data_list = self.net.format_data(_input_datas)
            rst_list = self.net(data_list)
            total_loss, info_list = self.net.compute_loss(data_list, rst_list)
            results["total_loss"] = total_loss.item()

            _info_list = []
            for info in info_list:
                if isinstance(info, list):
                    _info = [i.item() for i in info]
                else:
                    _info = info.item()

                _info_list.append(_info)
            results["info_list"] = _info_list

            total_loss.backward()
            self.optimizer.step()

            batch_duration = time.time() - batch_begin
            self.local_step += 1
            if self.local_step % self.config_manager.save_model_steps != 0:
                self.step_train_times.append(batch_duration)

            if self.is_chief_rank and (
                self.local_step == 0
                or self.local_step % self.config_manager.display_every == 0
            ):
                results["ip"] = self.config_manager.ips[0]
                results["batch_size"] = self.config_manager.batch_size
                results["step"] = self.local_step
                results["gpu_nums"] = self.rank_size
                results["sample_recv_speed"] = self.dataset.get_recv_speed()
                results["sample_consume_speed"] = self.get_sample_consume_speed(
                    self.config_manager.batch_size, self.step_train_times
                )
                self.log_manager.print_result(results)

            if (
                self.local_step % self.config_manager.save_model_steps == 0
                and self.is_chief_rank
            ):
                self.save_checkpoint(self.config_manager.save_model_dir)
                _, msg = self.model_manager.send_model(
                    self.config_manager.save_model_dir,
                    self.config_manager.send_model_dir,
                )
                self.log_manager.print_info(msg)

        images_per_sec = (
            (time.time() - start_time)
            / (self.config_manager.max_steps - self.config_manager.warmup_steps)
            * self.config_manager.batch_size
        )
        self.log_manager.print_info("-" * 64)
        self.log_manager.print_info("total images/sec: %.2f" % images_per_sec)
        self.log_manager.print_info("-" * 64)
        # Save the model checkpoint.
        if self.is_chief_rank:
            self.save_checkpoint(self.config_manager.save_model_dir)
            self.model_manager.send_model(
                self.config_manager.save_model_dir, self.config_manager.send_model_dir
            )

    def run(self):
        self._do_train()

    def save_checkpoint(self, checkpoint_dir: str):
        checkpoint_file = os.path.join(checkpoint_dir, "model.pth")
        if not self.is_chief_rank:
            return  # only save checkpoint on master node
        torch.save(
            {
                "network_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_file,
        )

    def load_checkpoint(self, checkpoint_file: str):
        if not self.is_chief_rank:
            return  # only load checkpoint on master node and then broadcast
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))

        self.net.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def get_sample_consume_speed(self, batch_size, step_train_times, scale=1):
        if not step_train_times:
            return ""
        if len(step_train_times) <= 1:
            return step_train_times[0]
        times = np.array(step_train_times[1:])
        speed_mean = scale * batch_size / np.mean(times)
        return speed_mean
