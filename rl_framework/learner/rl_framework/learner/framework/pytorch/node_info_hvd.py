#  definition of node and gradient fusion.
try:
    import horovod.torch as hvd
    hvd.init()
    has_hvd = True
except Exception:  # pylint: disable=broad-except
    has_hvd = False


#  The class describes some information about the node
class NodeInfo(object):
    #  The constructor. Initialization of node details
    def __init__(self):
        self.has_hvd = has_hvd
        if has_hvd:
            #  The sequence number of the training process in all nodes
            self.rank = hvd.rank()
            #  Total number of training nodes
            self.size = hvd.size()
            #  The sequence number of the training process in local node
            self.local_rank = hvd.local_rank()
            #  Total number of local training nodes
            self.local_size = hvd.local_size()
        else:
            self.rank = 0
            self.size = 1
            self.local_rank = 0
            self.local_size = 1
