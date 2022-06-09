import configparser
import json
from rl_framework.learner.framework.common import *


@singleton
class ConfigControl(object):
    def __init__(
        self, config_file="code/common.conf", rank=0, size=1, local_rank=0, local_size=1
    ):
        self.hvd_rank = rank
        self.hvd_size = size
        self.hvd_local_rank = local_rank
        self.hvd_local_size = local_size
        self.data_keys = ["input_data"]
        config = configparser.ConfigParser()
        config.read(config_file)
        self.ips = config.get("main", "ips").split(",")
        self.ports = json.loads(config.get("main", "ports"))
        self.batch_size = config.getint("main", "batch_size")
        self.mem_process_num = config.getint("main", "mem_process_num")
        self.save_model_steps = config.getint("main", "save_model_steps")
        self.display_every = config.getint("main", "display_every")
        self.max_steps = config.getint("main", "max_steps")
        self.training_type = config.get("main", "training_type")
        self.train_dir = config.get("main", "train_dir")
        self.warmup_steps = (
            config.get("main", "warmup_step")
            if "warmup_step" in config.options("main")
            else 0
        )
        self.mempool_path = (
            config.get("main", "mempool_path")
            if "mempool_path" in config.options("main")
            else "/data1/reinforcement_platform/mem_pool_server_pkg"
        )
        self.print_timeline = (
            config.getboolean("main", "print_timeline")
            if "print_timeline" in config.options("main")
            else False
        )
        self.print_variables = (
            config.getboolean("main", "print_variables")
            if "print_variables" in config.options("main")
            else False
        )

        self.use_init_model = config.getboolean("model", "use_init_model")
        self.use_xla = config.getboolean("model", "use_xla")
        self.use_mix_precision = config.getboolean("model", "use_mix_precision")
        self.use_fp16 = (
            config.getboolean("model", "use_fp16")
            if "use_fp16" in config.options("model")
            else False
        )
        self.save_path = (
            config.get("model", "save_path")
            if "save_path" in config.options("model")
            else "../send_model/model/"
        )
        self.push_to_modelpool = (
            config.get("model", "push_to_modelpool")
            if "push_to_modelpool" in config.options("model")
            else False
        )

        self.check_values = config.getboolean("grads", "check_values")
        self.use_fusion = config.getboolean("grads", "use_fusion")
        if self.use_fusion:
            self.piecewise_fusion_schedule = config.get(
                "grads", "piecewise_fusion_schedule"
            )
        else:
            self.piecewise_fusion_schedule = None
        self.use_xla_fusion = (
            config.getboolean("grads", "use_xla_fusion")
            if "use_xla_fusion" in config.options("grads")
            else False
        )
        self.use_grad_clip = config.getboolean("grads", "use_grad_clip")
        self.grad_clip_range = config.getfloat("grads", "grad_clip_range")
        self.sparse_as_dense = config.getboolean("grads", "sparse_as_dense")
        self.grad_to_fp16 = config.getboolean("grads", "to_fp16")

        self.max_sample = config.getint("dataset", "store_max_sample")
        self.sample_process = config.getint("dataset", "sample_process")
        self.batch_process = config.getint("dataset", "batch_process")
