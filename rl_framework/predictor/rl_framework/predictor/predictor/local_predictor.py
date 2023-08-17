# -*- coding: utf-8 -*-

import os

import tensorflow as tf

from rl_framework.predictor.predictor.base_predictor import BasePredictor


class LocalTFPredictor(BasePredictor):
    """An LocalTFPredictor object is used to perform model loading and
    inference operations for tensorflow models.
    None of the methods are thread safe. The object is intended to be used
    by a single thread and simultaneously calling different methods
    with different threads is not supported and will cause undefined
    behavior.

    Parameters
    ----------
    graph: tf.Graph
        A tf.Graph object that contains a set of tf.Operation and
        tf.Tensor objects.
    model_pool_addr: list of str
        The address of model_pool, e.g. ['localhost:10013:10014']
    cpu_num: int
        cpu number for tf session config
    """

    def __init__(self, graph, cpu_num):
        super().__init__()
        # init session
        self._graph = graph
        self._sess_config = tf.ConfigProto(
            device_count={"CPU": cpu_num},
            inter_op_parallelism_threads=cpu_num,
            intra_op_parallelism_threads=cpu_num,
            log_device_placement=False,
        )
        self._sess = tf.Session(graph=self._graph, config=self._sess_config)

    def load_model(self, model_path):
        """Request the model_pool to get the specified model file,
        then load or reload the model.

        Parameters
        ----------
        model_key : str
            The name of the model to be loaded.

        Returns
        ------
        bool
            The result of load_model.

        """
        return self._tf_load_api(model_path)

    def inference(self, input_list, output_list):
        """Use tf.Session.run to run TensorFlow operations.
        Feed tensors in 'input_list' and evaluate tensors in 'output_list'.

        Parameters
        ----------
        input_list : list of InferInput
            The list of input tensors.
        output_list : list of InferOutput
            The list of output tensors.

        Returns
        ------
        output_list
            The list of output tensors.

        """

        input_names = [inp.name for inp in input_list]
        input_datas = [inp.data for inp in input_list]
        feed_dict = dict(zip(input_names, input_datas))
        output_names = [output.name for output in output_list]
        output_datas = self._sess.run(output_names, feed_dict=feed_dict)
        for output, data in zip(output_list, output_datas):
            output.data = data
        return output_list

    def _tf_load_api(self, model_path):
        raise NotImplementedError


class LocalCkptPredictor(LocalTFPredictor):
    """An LocalCkptPredictor object is used to perform model loading and
    inference operations for tensorflow models saved as checkpoint.

    Parameters
    ----------
    graph: tf.Graph
        A tf.Graph object that contains a set of tf.Operation and
        tf.Tensor objects.
    model_pool_addr: list of str
        The address of model_pool, e.g. ['localhost:10013:10014']
    cpu_num: int
        cpu number for tf session config
    ckpt_name: str
        The prefix of model file, default as 'model.ckpt'
    """

    def __init__(self, graph, cpu_num=1, ckpt_name="model.ckpt"):
        super().__init__(graph, cpu_num)
        self._ckpt_name = ckpt_name
        with self._graph.as_default():
            self._saver = tf.train.Saver(tf.global_variables())

    def _tf_load_api(self, model_path):
        """Load checkpoint.

        Parameters
        ----------
        model_dir : str
            The path of checkpoint.
        """
        ckpt_path = os.path.join(model_path, self._ckpt_name)
        self._saver.restore(self._sess, ckpt_path)
        return True
