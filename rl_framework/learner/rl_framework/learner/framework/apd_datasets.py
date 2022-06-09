import tensorflow as tf
from rl_framework.learner.framework.common.config_control import ConfigControl


class Datasets(object):
    def __init__(self, NetworkDataset, adapter):
        config_manager = ConfigControl()
        self.dataset = NetworkDataset(config_manager, adapter)

    def next_batch(self):
        with tf.name_scope("batch_input"):
            return self.dataset.get_next_batch()

    def get_recv_speed(self):
        return self.dataset.get_recv_speed()
