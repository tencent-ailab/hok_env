import os
import sys

os.environ["dataop"] = os.getcwd() + "/code/shm_lib/"
sys.path.append("/code/code/common/")
from algorithm import Algorithm
from networkmodel.offline_rlinfo_adapter import OfflineRlInfoAdapter

from rl_framework.learner import Trainer
from trainer_slow import BenchmarkSlow
from rl_framework.learner.dataset.network_dataset.network_dataset_zmq_dataset import (
    NetworkDataset,
)

if __name__ == "__main__":

    config_path = os.getcwd() + "/code/common.conf"
    network = Algorithm()
    trainer = Trainer(
        network,
        NetworkDataset,
        OfflineRlInfoAdapter,
        config_path,
        BenchmarkClass=BenchmarkSlow,
    )
    trainer.run()
