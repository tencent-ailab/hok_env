#  @package rl_framework.learner.dataset.sample_generation
#  The module defines base class for sample parsing
from abc import abstractmethod


#  The class is the base class for sample parsing
class OfflineRlInfoAdapter(object):
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
    #  get_data_shapes() is a static function.
    #  @return [[sample_len]]
    @staticmethod
    def get_data_shapes():
        raise NotImplementedError("deserialization: not implemented")
