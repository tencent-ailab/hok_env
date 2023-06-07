import os

from rl_framework.learner.dataset.sample_generation.offline_rlinfo_adapter import (
    OfflineRlInfoAdapter,
)
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.learner.framework.common.log_manager import LogManager

from config.Config import Config

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("single_test", 0, "test_mode")


def main(argv):
    config_path = os.path.join(os.path.dirname(__file__), "config", "common.conf")

    config_manager = ConfigControl(config_path)
    os.makedirs(config_manager.save_model_dir, exist_ok=True)
    os.makedirs(config_manager.train_dir, exist_ok=True)
    os.makedirs(config_manager.send_model_dir, exist_ok=True)

    use_backend = config_manager.backend

    if use_backend == "tensorflow":
        from rl_framework.learner.framework.tensorflow.model_manager import ModelManager
        from rl_framework.learner.dataset.network_dataset.tensorflow.network_dataset_zmq import (
            NetworkDataset,
        )
        from rl_framework.learner.dataset.network_dataset.tensorflow.network_dataset_random import (
            NetworkDataset as NetworkDatasetRandom,
        )
        from rl_framework.learner.framework.tensorflow.apd_benchmark import Benchmark
        from rl_framework.learner.framework.tensorflow.gradient_fusion import NodeInfo
        from networkmodel.tensorflow.NetworkModel import NetworkModel
    elif use_backend == "pytorch":
        from rl_framework.learner.framework.pytorch.node_info import NodeInfo
        from rl_framework.learner.framework.pytorch.model_manager import ModelManager
        from rl_framework.learner.dataset.network_dataset.pytorch.network_dataset_zmq import (
            NetworkDataset,
        )
        from rl_framework.learner.dataset.network_dataset.pytorch.network_dataset_random import (
            NetworkDataset as NetworkDatasetRandom,
        )
        from rl_framework.learner.framework.pytorch.apd_benchmark import Benchmark
        from networkmodel.pytorch.NetworkModel import NetworkModel
    else:
        raise NotImplementedError(
            "Support backend in [pytorch, tensorflow], Check your training backend..."
        )

    adapter = OfflineRlInfoAdapter(Config.data_shapes)
    if FLAGS.single_test:
        dataset = NetworkDatasetRandom(config_manager, adapter)
        config_manager.push_to_modelpool = False
    else:
        dataset = NetworkDataset(config_manager, adapter)

    node_info = NodeInfo()
    log_manager = LogManager()
    network = NetworkModel()
    model_manager = ModelManager(config_manager.push_to_modelpool)
    benchmark = Benchmark(
        network,
        dataset,
        log_manager,
        model_manager,
        config_manager,
        node_info,
    )
    benchmark.run()


if __name__ == "__main__":
    from absl import app as absl_app

    absl_app.run(main)
