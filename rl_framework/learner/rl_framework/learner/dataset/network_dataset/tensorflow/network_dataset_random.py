import numpy as np
import tensorflow as tf
from rl_framework.learner.dataset.network_dataset import NetworkDatasetBase


class NetworkDataset(NetworkDatasetBase):
    def __init__(self, config_manager, adapter):
        self.use_fp16 = config_manager.use_fp16
        self.batch_size = config_manager.batch_size
        self.data_shapes = adapter.get_data_shapes()
        self.sample_length = self.data_shapes[0][0]
        self.sample = np.random.random([self.batch_size, self.sample_length])
        if self.use_fp16:
            self.sample = self.sample.astype(np.float16)
            self.key_types = [tf.float16]
        else:
            self.sample = self.sample.astype(np.float32)
            self.key_types = [tf.float32]

    def get_next_batch(self):
        return tf.py_func(self.next_batch, [], self.key_types)

    def next_batch(self):
        return [self.sample]
