import os
import sys


# sys.path.append("/aiarena/code/") add common to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_framework.learner.dataset.sample_generation.offline_rlinfo_adapter import (
    OfflineRlInfoAdapter,
)

from rl_framework.learner.framework.common.log_manager import LogManager
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.common.logging import setup_logger

from absl import flags

config_path = os.path.join(os.path.dirname(__file__), "config", "common.conf")
train_log = "/aiarena/logs/learner/train.log"


def run(common_config, framework_config, single_test):
    """
    common_config: 模型配置
    framework_config: 框架配置, 为框架的ConfigControl
    single_test: 单独测试learner
    """

    config_manager = framework_config  # alias

    setup_logger(train_log)
    os.makedirs(config_manager.save_model_dir, exist_ok=True)
    os.makedirs(config_manager.train_dir, exist_ok=True)
    os.makedirs(config_manager.send_model_dir, exist_ok=True)

    training_backend = config_manager.backend

    if training_backend == "pytorch":
        from common.algorithm_torch import Algorithm
        from rl_framework.learner.dataset.network_dataset.pytorch.network_dataset_zmq import (
            NetworkDataset as NetworkDatasetZMQ,
        )
        from rl_framework.learner.dataset.network_dataset.pytorch.network_dataset_random import (
            NetworkDataset as NetworkDatasetRandom,
        )
        from rl_framework.learner.framework.pytorch.model_manager import ModelManager
        from rl_framework.learner.framework.pytorch.apd_benchmark import Benchmark

        distributed_backend = config_manager.distributed_backend
        if distributed_backend == "horovod":
            from rl_framework.learner.framework.pytorch.node_info_hvd import NodeInfo
        else:
            from rl_framework.learner.framework.pytorch.node_info_ddp import NodeInfo
    elif training_backend == "tensorflow":
        from common.algorithm_tf import Algorithm
        from rl_framework.learner.dataset.network_dataset.tensorflow.network_dataset_zmq import (
            NetworkDataset as NetworkDatasetZMQ,
        )
        from rl_framework.learner.dataset.network_dataset.tensorflow.network_dataset_random import (
            NetworkDataset as NetworkDatasetRandom,
        )
        from rl_framework.learner.framework.tensorflow.model_manager import ModelManager
        from rl_framework.learner.framework.tensorflow.apd_benchmark import Benchmark
        from rl_framework.learner.framework.tensorflow.gradient_fusion import NodeInfo
    else:
        raise NotImplementedError(
            "Support backend in [pytorch, tensorflow], Check your training backend..."
        )

    adapter = OfflineRlInfoAdapter(common_config.data_shapes)

    node_info = NodeInfo()
    network = Algorithm()

    if single_test:
        dataset = NetworkDatasetRandom(config_manager, adapter)
        config_manager.push_to_modelpool = False
    else:
        dataset = NetworkDatasetZMQ(
            config_manager, adapter, port=config_manager.ports[node_info.local_rank]
        )

    model_manager = ModelManager(config_manager.push_to_modelpool)
    benchmark = Benchmark(
        network,
        dataset,
        LogManager(),
        model_manager,
        config_manager,
        node_info,
        slow_time=common_config.slow_time,
    )
    benchmark.run()


def main(_):
    from common.config import Config

    FLAGS = flags.FLAGS

    config_manager = ConfigControl(config_path)
    config_manager.backend = Config.backend
    config_manager.use_init_model = Config.use_init_model

    run(Config, config_manager, FLAGS.single_test)


if __name__ == "__main__":
    from absl import app as absl_app

    flags.DEFINE_boolean("single_test", 0, "test_mode")

    absl_app.run(main)
