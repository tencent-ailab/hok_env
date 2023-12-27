import configparser
import json
from rl_framework.learner.framework.common import *


@singleton
class ConfigControl(object):
    def __init__(self, config_file="code/common.conf"):
        self.data_keys = ["input_data"]
        config = configparser.ConfigParser()
        config.read(config_file)

        self.backend = (
            config.get("main", "backend")
            if "backend" in config.options("main")
            else "tensorflow"
        )

        self.ips = config.get("main", "ips").split(",")
        self.ports = json.loads(config.get("main", "ports"))
        self.batch_size = config.getint("main", "batch_size")
        self.mem_process_num = config.getint("main", "mem_process_num")
        self.save_model_steps = config.getint("main", "save_model_steps")
        self.save_model_seconds = config.getint("main", "save_model_seconds", fallback=2 * 60)
        self.save_model_dir = (
            config.get("main", "save_model_dir")
            if "save_model_dir" in config.options("main")
            else "./checkpoints"
        )
        self.send_model_dir = (
            config.get("main", "send_model_dir")
            if "send_model_dir" in config.options("main")
            else "../send_model/model/"
        )
        self.push_to_modelpool = (
            config.getboolean("main", "push_to_modelpool")
            if "push_to_modelpool" in config.options("main")
            else False
        )
        self.display_every = config.getint("main", "display_every")
        self.max_steps = config.getint("main", "max_steps")
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

        # torch-only: horovod / ddp / None
        self.distributed_backend = config.get('main', 'distributed_backend', fallback=None)

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

        self.dump_profile = config.getboolean('main', 'dump_profile', fallback=False)  # torch-only

        self.use_init_model = config.getboolean("model", "use_init_model")
        self.load_optimizer_state = config.getboolean("model", "load_optimizer_state", fallback=True)
        self.init_model_path = (
            config.get("model", "init_model_path")
            if "init_model_path" in config.options("model")
            else "./model/init/"
        )
        self.use_xla = config.getboolean('model', 'use_xla')  # tf-only
        self.use_jit = config.getboolean('model', 'use_jit', fallback=False)  # torch-only
        self.use_compile = config.getboolean('model', 'use_compile', fallback=False)  # torch-only
        self.use_mix_precision = config.getboolean("model", "use_mix_precision")
        self.channels_last = config.getboolean('model', 'channels_last', fallback=False)  # torch-only
        self.use_fp16 = (
            config.getboolean("model", "use_fp16")
            if "use_fp16" in config.options("model")
            else False
        )

        self.has_unused_params = config.getboolean('model', 'has_unused_params', fallback=False)  # torch ddp-only

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
