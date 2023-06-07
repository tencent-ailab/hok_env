

class ModelSynBase(object):
    def __init__(self, address):
        raise NotImplementedError("__init__: not implemented!")

    def syn_model(self, model_path, model_key = None):
        raise NotImplementedError("syn_model: not implemented!")
