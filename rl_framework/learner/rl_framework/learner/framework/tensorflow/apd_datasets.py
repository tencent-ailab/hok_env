import tensorflow as tf


class Datasets(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def next_batch(self):
        with tf.name_scope("batch_input"):
            return self.dataset.get_next_batch()

    def get_recv_speed(self):
        return self.dataset.get_recv_speed()
