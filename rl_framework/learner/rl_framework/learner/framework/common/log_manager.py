import json
import logging
import logging.handlers
import math
import os
import time

LOG_FN = logging.getLogger("main").debug
from rl_framework.learner.framework.common import *
from rl_framework.monitor import InfluxdbMonitorHandler


@singleton
class LogManager(object):
    def __init__(self):
        # create logger
        logger = logging.getLogger("main")
        logger.setLevel(logging.DEBUG)

        # set formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # set file handler: file rotates with time
        if not os.path.exists("/logs/gpu_log/"):
            # os.makedirs('./log')
            os.system("mkdir /logs/gpu_log/")
        rf_handler = logging.handlers.TimedRotatingFileHandler(
            filename="/logs/gpu_log/train.log", when="H", interval=12, backupCount=1
        )
        rf_handler.setLevel(logging.DEBUG)
        rf_handler.setFormatter(formatter)

        # set console handler: display on the console
        console = logging.StreamHandler()
        console.setLevel(logging.ERROR)
        console.setFormatter(formatter)

        # monitor logger
        self.monitor_logger = logging.getLogger("monitor")
        self.monitor_logger.setLevel(logging.INFO)
        monitor_handler = InfluxdbMonitorHandler("127.0.0.1")
        monitor_handler.setLevel(logging.INFO)
        self.monitor_logger.addHandler(monitor_handler)

        logger.addHandler(rf_handler)
        logger.addHandler(console)
        loss_file_path = "/logs/gpu_log/loss.txt"
        self.loss_writer = open(loss_file_path, "wt")
        self.total_noise_scale = 0.0

    def print_result(self, results):
        local_step = results["step"]
        batch_size = results["batch_size"]
        gpu_nums = results["gpu_nums"]
        recv_speed = results["sample_recv_speed"]
        consume_speed = results["sample_consume_speed"]

        loss = results["total_loss"]
        noise_scale = results["noise_scale"]
        batch_noisescale = float(batch_size * gpu_nums) / noise_scale
        noise_scale_mean = 0

        log_str = ""
        log_str += "step: %i" % local_step
        log_str += " images/sec mean = %.1f" % consume_speed
        if recv_speed is not None and recv_speed > 0:
            log_str += " recv_sample/sec = %i" % recv_speed
        log_str += " total_loss: %s" % loss
        log_str += " noise_scale: %.2f" % noise_scale
        log_str += " batch_noisescale: %.2f" % batch_noisescale
        self.total_noise_scale += noise_scale
        noise_scale_mean = self.total_noise_scale / float(local_step)
        log_str += " mean noise scale = %f" % (noise_scale_mean)
        LOG_FN(log_str)

        monitor_data = {}
        monitor_data["step"] = int(local_step)
        self._add_float(
            monitor_data, "sample_consumption_rate", consume_speed * 60 * gpu_nums
        )
        self._add_float(monitor_data, "total_loss", loss)
        self._add_float(monitor_data, "noise_scale", noise_scale)
        self._add_float(monitor_data, "batch_noisescale", batch_noisescale)
        self._add_float(monitor_data, "noise_scale_mean", noise_scale_mean)
        if "info_map" in results.keys():
            for k, v in results["info_map"].items():
                if type(v).__name__ == "list":
                    for index, data in enumerate(v):
                        self._add_float(monitor_data, f"{k}_{index}", data)
                elif type(v).__name__ == "ndarray":
                    results["info_map"][k] = v.tolist()
                    for index, data in enumerate(results["info_map"][k]):
                        self._add_float(monitor_data, f"{k}_{index}", data)
                else:
                    self._add_float(monitor_data, k, v)
        else:
            for k, v in results["info_list"].items():
                if type(v).__name__ == "list":
                    for index, data in enumerate(v):
                        self._add_float(monitor_data, f"{k}_{index}", data)
                elif type(v).__name__ == "ndarray":
                    results["info_list"][k] = v.tolist()
                    for index, data in enumerate(results["info_list"][k]):
                        self._add_float(monitor_data, f"{k}_{index}", data)
                else:
                    self._add_float(monitor_data, k, v)
        if recv_speed is not None and recv_speed > 0:
            monitor_data["sample_generation_rate"] = float(recv_speed * 60 * gpu_nums)
        self.upload_monitor_data(monitor_data)

        if "train_has_inf_nan" in results:
            log_str = "train_has_inf_nan of step %i:  " % local_step
            for info in results["train_has_inf_nan"]:
                log_str += "%s, " % str(info)
            LOG_FN(log_str)

        if "info_map" in results.keys() or "info_list" in results.keys():
            self._write_loss_log(results)

    def _add_float(self, data, key, val):
        val = float(val)
        if not math.isnan(val) and not math.isinf(val):
            data[key] = val

    def _write_loss_log(self, results):
        local_step = results["step"]
        hostname = results["ip"]
        timestamp = time.strftime("%m/%d/%Y-%H:%M:%S", time.localtime())
        info_map = {}
        if "info_map" in results.keys():
            info_map = results["info_map"]
        else:
            info_map = results["info_list"]

        loss_log = {
            "role": "learner",
            "ip_address": str(hostname),
            "step": str(local_step),
            "timestamp": timestamp,
            "info_map": str(info_map),
        }
        for key, val in sorted(loss_log.items()):
            if hasattr(val, "dtype"):
                loss_log[key] = float(val)
        self.loss_writer.write(json.dumps(loss_log) + "\n")
        self.loss_writer.flush()

    def print_info(self, info):
        LOG_FN(info)

    def upload_monitor_data(self, data: dict):
        self.monitor_logger.info(data)
