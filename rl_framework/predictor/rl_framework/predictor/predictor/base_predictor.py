# -*- coding: utf-8 -*-


class BasePredictor(object):
    """The BasePredictor class is an abstract base class."""

    def __init__(self):
        pass

    def load_model(self, model_name):
        raise NotImplementedError

    def inference(self, input_list, output_list):
        raise NotImplementedError
