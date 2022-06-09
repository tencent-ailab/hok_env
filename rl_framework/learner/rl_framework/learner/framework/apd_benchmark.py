# -*- coding: utf-8 -*-
import time

# import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from rl_framework.learner.framework.apd_datasets import Datasets
from rl_framework.learner.framework.apd_model import Graphs
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.learner.framework.common.gradient_fusion import NodeInfo
from rl_framework.learner.framework.common.log_manager import LogManager
from rl_framework.learner.framework.common.model_manager import ModelManager
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.util import nest


def create_config_proto(gpu_index):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 2
    config.inter_op_parallelism_threads = 0
    config.gpu_options.force_gpu_compatible = True
    config.gpu_options.visible_device_list = str(gpu_index)
    config.graph_options.rewrite_options.arithmetic_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF
    )
    return config


class Benchmark(object):
    def __init__(
        self,
        network,
        dataset,
        adapter,
        config_path,
        LogManagerClass=LogManager,
        ModelManagerClass=ModelManager,
    ):
        self.log_manager = LogManagerClass()
        self.log_manager.print_info("init starting")
        self.node_info = NodeInfo()
        self.config_manager = ConfigControl(
            config_path,
            self.node_info.rank,
            self.node_info.size,
            self.node_info.local_rank,
            self.node_info.local_size,
        )
        self.is_chief_rank = self.node_info.rank == 0
        self.graph = Graphs(network)
        self.dataset = Datasets(dataset, adapter)
        self.local_step = 0
        self.step_train_times = list()
        self.total_noise_scale = 0.0
        self.noise_scale_times = 0
        self.skip_update_times = 0
        self.ModelManagerClass = ModelManagerClass
        self.log_manager.print_info("init finished")

    def _build_model(self):
        with tf.device("/cpu:0"):
            input_datas = self.dataset.next_batch()

        (self.enqueue_ops, self.fetches) = self.graph.build_model(input_datas)

        fetches_list = nest.flatten(list(self.fetches.values()))
        main_fetch_group = tf.group(*fetches_list)
        with tf.device("/cpu:0"):
            self.global_step = tf.train.get_global_step()
            self.fetches["global_step"] = self.global_step
            with tf.control_dependencies([main_fetch_group]):
                self.fetches["inc_global_step"] = self.global_step.assign_add(1)

    def _init_model(self):
        self.model_manager = self.ModelManagerClass(
            self.config_manager.push_to_modelpool
        )
        bcast_global_variables_op = self.node_info.get_bcast_op()
        local_var_init_op = tf.local_variables_initializer()
        local_save_model_secs = 0

        self.log_manager.print_info("local_save_model_secs: %d" % local_save_model_secs)
        self.sv = tf.train.Supervisor(
            is_chief=True,
            logdir=self.config_manager.train_dir,
            ready_for_local_init_op=None,
            local_init_op=local_var_init_op,
            saver=None,
            global_step=self.global_step,
            summary_op=None,
            summary_writer=None,
            save_model_secs=local_save_model_secs,
        )

        self.sess = self.sv.prepare_or_wait_for_session(
            master="",
            config=create_config_proto(self.node_info.local_rank),
            start_standard_services=True,
        )

        if self.config_manager.use_init_model:
            self.model_manager.restore_model(self.sess)
        if self.config_manager.print_variables:
            self.model_manager.print_variables(self.sess)

        if bcast_global_variables_op:
            self.sess.run(bcast_global_variables_op)

        if self.is_chief_rank:
            self.model_manager.save_model(self.sess, self.config_manager.save_path)
        self.sess.run(self.enqueue_ops)
        self.init_global_step = self.sess.run(self.global_step)

    def _do_train(self):

        self.log_manager.print_info("Start training...")
        start_time = time.time()
        for _ in range(self.config_manager.warmup_steps, self.config_manager.max_steps):
            batch_begin = time.time()
            if (
                self.is_chief_rank
                and self.local_step == 100
                and self.config_manager.print_timeline
            ):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                results = self.sess.run(
                    self.fetches, options=run_options, run_metadata=run_metadata
                )
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open("./log/timeline.json", "w") as f_timeline:
                    f_timeline.write(ctf)
            else:
                results = self.sess.run(self.fetches)

            batch_duration = time.time() - batch_begin
            self.local_step += 1
            if self.local_step % self.config_manager.save_model_steps != 0:
                self.step_train_times.append(batch_duration)

            # if (self.node_info.local_rank == self.node_info.local_size - 1) and \
            if self.is_chief_rank and (
                self.local_step == 0
                or self.local_step % self.config_manager.display_every == 0
            ):
                # if (self.local_step >= 0 and self.is_chief_rank and \
                #    (self.local_step == 0 or self.local_step % self.config_manager.display_every == 0)):
                results["ip"] = self.config_manager.ips[0]
                results["batch_size"] = self.config_manager.batch_size
                results["step"] = self.local_step
                results["gpu_nums"] = self.node_info.size
                results["sample_recv_speed"] = self.dataset.get_recv_speed()
                results["sample_consume_speed"] = self.get_sample_consume_speed(
                    self.config_manager.batch_size, self.step_train_times
                )
                self.log_manager.print_result(results)

            if (
                self.local_step % self.config_manager.save_model_steps == 0
                and self.is_chief_rank
            ):
                _, msg = self.model_manager.save_model(
                    self.sess, self.config_manager.save_path
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
            self.model_manager.save_model(self.sess, self.config_manager.save_path)
        self.sv.stop()

    def _benchmark_run(self):
        self._build_model()
        self._init_model()
        self._do_train()

    def run(self):
        with tf.Graph().as_default():
            self._benchmark_run()

    def get_sample_consume_speed(self, batch_size, step_train_times, scale=1):
        if not step_train_times:
            return ""
        if len(step_train_times) <= 1:
            return step_train_times[0]
        times = np.array(step_train_times[1:])
        speed_mean = scale * batch_size / np.mean(times)
        return speed_mean
