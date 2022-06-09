#  @package rl_framework.learner.algorithms.base
#  The module defines some base classes, which includes model and algorithm
from abc import abstractmethod


#  The class is the base class which defines the implementation of the algorithm
class Algorithm(object):
    #  The constructor.
    #  @param self The object pointer.
    #  @param model The object defines the structure of the network;
    #         Its type is rl_framework.learner.algorithms.base.Model
    def __init__(self, model):
        #  defines the structure of the network
        self.model = model

    #  Provides the interface for constructing tensorflow graph,
    #  build_graph() is a virtual function.
    #  @param self The object pointer.
    #  @param datas  Training data of network,Type needs to be tensorflow.op
    #  @param update  Represents the current number of steps that have been iterated
    #  @return loss loss of local iteration
    #  @return other_info The object is used to store the information the user wants to print,
    #          In the sample code, we store the accuracy.
    @abstractmethod
    def build_graph(self, datas, update):
        raise NotImplementedError("build_graph: not implemented")
        # return loss, other_info

    #  Provides an interface to the optimizer, get_optimizer() is a virtual function.
    #  @param self The object pointer.
    #  @return Returns the optimizer defined in tensorflow.
    @abstractmethod
    def get_optimizer(self):
        raise NotImplementedError("get optimizer: not implemented")
