import numpy as np
from config.config import ModelConfig as Config
from rl_framework.learner.dataset.sample_generation.offline_rlinfo_adapter import (
    OfflineRlInfoAdapter as OfflineRlInfoAdapterBase,
)


class OfflineRlInfoAdapter(OfflineRlInfoAdapterBase):
    def __init__(self):
        super().__init__()

    def deserialization(self, receive_data):
        return self.deserialization_bytes(receive_data)

    def deserialization_bytes(self, receive_data):
        data = []
        data.append(np.frombuffer(receive_data, "f4"))
        return data

    @staticmethod
    def get_data_shapes():
        data_len = 0
        for value in Config.data_shapes:
            data_len += value[0]
        return [[data_len]]
