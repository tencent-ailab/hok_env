from abc import abstractmethod


class NetworkDatasetBase(object):
    def __init__(self, config_manager, adapter):
        raise NotImplementedError("build model: not implemented!")

    @abstractmethod
    def get_next_batch(self):
        raise NotImplementedError("build model: not implemented!")

    def get_recv_speed(self):
        return None
