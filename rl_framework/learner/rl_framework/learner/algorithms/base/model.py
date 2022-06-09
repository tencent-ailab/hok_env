#  @package rl_framework.learner.algorithms.base
#  The module defines some base classes, which includes model and algorithm
from abc import abstractmethod

#  The class is the base class which defines the structure of the network


class Model(object):
    #  The constructor.
    #  @param self The object pointer.
    def __init__(self, *args):
        pass

    #  This function provides the interface defined by the network structure,
    #  inference() is a virtual function.
    #  @param feature Input data of network prediction
    @abstractmethod
    def inference(self, feature):
        raise NotImplementedError("build model: not implemented!")
