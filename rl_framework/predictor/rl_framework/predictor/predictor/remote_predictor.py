# -*- coding: utf-8 -*-
import logging
import random
import time

from rl_framework.predictor.predictor.infer_input_output import InferInput, InferOutput
from rl_framework.predictor.predictor.base_predictor import BasePredictor
from rl_framework.predictor.inference_server.inference_server_apis import (
    InferenceServerAPIs,
)


class RemotePredictor(BasePredictor):
    """An RemotePredictor object is used to perform model loading and
    inference operations with the remote InferenceServer.
    None of the methods are thread safe. The object is intended to be used
    by a single thread and simultaneously calling different methods
    with different threads is not supported and will cause undefined
    behavior.

    Parameters
    ----------
    infer_server_addrs: list of str
        The address of the remote InferenceServer in the following format:
        [host:infer_port:triton_port;...], e.g. '[localhost:18000:8000;...]'
    """

    def __init__(self, infer_server_addrs):
        super().__init__()
        self._infer_server_addrs = infer_server_addrs
        self._inference_server_apis = None
        self._model_name = None
        self._init_infer_server_apis()

    def _init_infer_server_apis(self):
        """Random shuffle infer_server_addrs and choose one of them."""

        try:
            random.shuffle(self._infer_server_addrs)
            addr = self._infer_server_addrs[-1]
            ip, port1, port2 = addr.split(":")
            logging.info("Random choose one of infer_server_addrs, %s" % addr)
        except Exception:
            logging.error(
                "Wrong format of infer_server_addrs, %s" % self._inference_server_apis
            )
            raise

        url = "%s:%s" % (ip, port1)
        triton_url = "%s:%s" % (ip, port2)

        while True:  # repeat until succ
            try:
                self._inference_server_apis = InferenceServerAPIs(url, triton_url)
                break
            except Exception:  # pylint: disable=broad-except
                logging.error("Can not connect to address %s, retry..." % addr)
                time.sleep(1)

    def load_model(self, model_name):
        """Request the remote InferenceServer to load or reload the specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.

        Returns
        ------
        bool
            The result of load_model.

        """

        self._model_name = model_name
        while True:
            try:
                ret = self._inference_server_apis.load_model(model_name)
                break
            except Exception as e:  # pylint: disable=broad-except
                logging.error("load_model raise exception, retry...\nexception:%s" % e)
                time.sleep(1)
        return ret

    def inference(self, input_list, output_list, model_name=None):
        """Request the remote TritonServer to inference using the specified model.
        Set model name as self._model_name if 'model_name' is None.
        Feed tensors in 'input_list' and evaluate tensors in 'output_list'.

        Parameters
        ----------
        input_list : list of InferInput
            The list of input tensors.
        output_list : list of InferOutput
            The list of output tensors.
        model_name : str
            The name of model to be used.
        Returns
        ------
        output_list
            The list of output tensors.

        """

        if model_name is None:
            model_name = self._model_name
        while True:
            try:
                ret = self._inference_server_apis.inference(
                    input_list, output_list, model_name
                )
                break
            except Exception as e:  # pylint: disable=broad-except
                logging.error("inference raise exception, retry...\nexception:%s" % e)
                time.sleep(1)
        return ret
