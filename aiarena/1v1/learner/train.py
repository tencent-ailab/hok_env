import os
import sys

sys.path.append("/aiarena/code/common/")

from algorithm import Algorithm

from config.config import ModelConfig as Config
from common_config import ModelConfig as CommonConfig

from rl_framework.learner.dataset.sample_generation.offline_rlinfo_adapter import (
    OfflineRlInfoAdapter,
)

from benchmark_slow import BenchmarkSlow
from rl_framework.learner.dataset.network_dataset.tensorflow.network_dataset_zmq import (
    NetworkDataset,
)
from rl_framework.learner.dataset.network_dataset.tensorflow.network_dataset_random import (
    NetworkDataset as NetworkDatasetRandom,
)

from rl_framework.learner.framework.common.log_manager import LogManager
from rl_framework.learner.framework.tensorflow.model_manager import ModelManager
from rl_framework.learner.framework.tensorflow.gradient_fusion import NodeInfo
from rl_framework.learner.framework.common.config_control import ConfigControl

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("single_test", 0, "test_mode")


def main(argv):
    adapter = OfflineRlInfoAdapter(Config.data_shapes)

    config_path = os.path.join(os.path.dirname(__file__), "config", "common.conf")
    node_info = NodeInfo()
    config_manager = ConfigControl(
        config_path,
        node_info.rank,
        node_info.size,
        node_info.local_rank,
        node_info.local_size,
    )
    CommonConfig.BATCH_SIZE = config_manager.batch_size
    network = Algorithm()

    if FLAGS.single_test:
        dataset = NetworkDatasetRandom(config_manager, adapter)
        config_manager.push_to_modelpool = False
    else:
        dataset = NetworkDataset(config_manager, adapter)

    model_manager = ModelManager(config_manager.push_to_modelpool)
    benchmark = BenchmarkSlow(
        network,
        dataset,
        LogManager(),
        model_manager,
        config_manager,
        node_info,
    )
    benchmark.run()


if __name__ == "__main__":
    from absl import app as absl_app

    absl_app.run(main)
