#  @package rl_framework.learner.dataset.sample_generation
#  The module defines base class for sample parsing
from abc import abstractmethod
import numpy as np


#  The class is the base class for sample parsing
class OfflineRlInfoAdapterBase(object):
    #  The constructor.
    #  @param self The object pointer.
    def __init__(self):
        pass

    #  Sample production interface,
    #  deserialization() is a virtual function.
    #  @param self The object pointer.
    #  @param receive_data This object needs to contain the data
    #         to generate batch_size samples
    #  @return sampes
    @abstractmethod
    def deserialization(self, receive_data):
        raise NotImplementedError("deserialization: not implemented")

    #  Sample length acquisition interface,
    #  get_data_shapes() is a virtual function.
    #  @return [[sample_len]]
    @abstractmethod
    def get_data_shapes():
        raise NotImplementedError("deserialization: not implemented")


class OfflineRlInfoAdapter(OfflineRlInfoAdapterBase):
    def __init__(self, data_shapes):
        super().__init__()
        self.data_shapes = data_shapes

    def deserialization(self, receive_data):
        return self.deserialization_bytes(receive_data)

    def deserialization_bytes(self, receive_data):
        data = []
        data.append(np.frombuffer(receive_data, "f4"))
        return data

    def get_data_shapes(self):
        return [[sum(map(sum, self.data_shapes))]]
