import time
import tensorflow as tf
from rl_framework.learner.framework.apd_benchmark import Benchmark as BenchmarkBase
from common_config import Config

tf.logging.set_verbosity(tf.logging.ERROR)
from rl_framework.learner.framework.common.log_manager import LogManager
from rl_framework.learner.framework.common.model_manager import ModelManager
from tensorflow.python.client import timeline


class BenchmarkSlow(BenchmarkBase):
    def __init__(
        self,
        network,
        dataset,
        adapter,
        config_path,
        LogManagerClass=LogManager,
        ModelManagerClass=ModelManager,
    ):
        super(BenchmarkSlow, self).__init__(
            network,
            dataset,
            adapter,
            config_path,
            LogManagerClass=LogManager,
            ModelManagerClass=ModelManager,
        )
        self.slow_time = Config.slow_time

    def _do_train(self):

        self.log_manager.print_info("Start training...")
        start_time = time.time()
        for _ in range(self.config_manager.warmup_steps, self.config_manager.max_steps):
            batch_begin = time.time()
            # ===== sleep at here =====
            time.sleep(self.slow_time)
            # ===== sleep at here =====
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
