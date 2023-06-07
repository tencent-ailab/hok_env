import numpy as np
from rl_framework.learner.dataset.network_dataset import NetworkDatasetBase


class NetworkDataset(NetworkDatasetBase):
    def __init__(self, config_manager, adapter):
        self.use_fp16 = config_manager.use_fp16
        self.batch_size = config_manager.batch_size
        self.adapter = adapter
        self.data_shapes = self.adapter.get_data_shapes()
        self.sample_length = self.data_shapes[0][0]
        self.sample = np.random.random([self.batch_size, self.sample_length])
        if self.use_fp16:
            self.sample = self.sample.astype(np.float16)
        else:
            self.sample = self.sample.astype(np.float32)

    def get_next_batch(self):
        return self.sample
