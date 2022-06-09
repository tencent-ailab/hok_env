import os
import sys

import tensorflow as tf

try:
    import horovod.tensorflow as hvd

    has_hvd = True
except Exception:  # pylint: disable=broad-except
    has_hvd = False
from rl_framework.learner.framework.apd_benchmark import Benchmark as BenchmarkBase
from rl_framework.learner.framework.common.log_manager import LogManager
from rl_framework.learner.framework.common.model_manager import ModelManager

tf.logging.set_verbosity(tf.logging.ERROR)


class Trainer(object):
    def __init__(
        self,
        network,
        DataSetClass,
        AdapterClass,
        config_path,
        LogManagerClass=LogManager,
        ModelManagerClass=ModelManager,
        BenchmarkClass=BenchmarkBase,
    ):
        if has_hvd:
            hvd.init()
        self.bench = BenchmarkClass(
            network,
            DataSetClass,
            AdapterClass,
            config_path,
            LogManagerClass,
            ModelManagerClass,
        )

    def run(self):
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
        os.environ["TF_AUTOTUNE_THRESHOLD"] = "1"
        self.bench.run()
