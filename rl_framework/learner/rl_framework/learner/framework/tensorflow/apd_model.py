# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.learner.framework.tensorflow.gradient_fusion import GradientFusion
from tensorflow.python.ops import data_flow_ops


class Graphs(object):
    def __init__(self, network):
        self.config_manager = ConfigControl()
        self.batch_size = self.config_manager.batch_size
        self.cpu_device = "/cpu:0"
        # if tf.test.is_gpu_available():
        if tf.test.is_built_with_cuda():
            self.device = "/gpu:0"
        else:
            self.device = self.cpu_device
        if self.config_manager.use_fp16:
            self.key_types = [tf.float16]
        else:
            self.key_types = [tf.float32]
        self.network = network
        self.gradient_fusion = GradientFusion()
        self.loss_has_inf_nan = None
        self.grad_has_inf_nan = None

    def get_data_list_shape(self, data_list):
        list_shapes = []
        for i in range(len(data_list)):
            list_shapes.append(data_list[i].get_shape())
        return list_shapes

    def build_model(self, input_datas):
        enqueue_ops = list()
        fetches = dict()
        training_ops = list()

        with tf.device(self.cpu_device):
            global_step = tf.train.get_or_create_global_step()
            self.global_step = global_step
            datas = input_datas

        with tf.variable_scope("", reuse=tf.AUTO_REUSE), tf.name_scope(
            "tower_0"
        ) as name_scope:
            with tf.xla.experimental.jit_scope(self.config_manager.use_xla):
                (
                    loss,
                    info_list,
                    gradvars,
                    max_noisescale,
                    gpu_copy_stage_op,
                    gpu_compute_stage_op,
                ) = self._add_forward_pass_and_gradients(datas)

            enqueue_ops.append(gpu_copy_stage_op)
            enqueue_ops.append(gpu_compute_stage_op)

            fetches["enqueue_ops"] = enqueue_ops
            fetches["info_list"] = info_list
            fetches["noise_scale"] = max_noisescale

            with tf.device(self.device):
                if not self.config_manager.use_mix_precision:
                    self.opt = self.network.get_optimizer()

                training_ops.append([self.opt.apply_gradients(gradvars)])
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                train_op = tf.group(*(training_ops + update_ops))

            fetches["train_op"] = train_op
            fetches["total_loss"] = loss
            if self.grad_has_inf_nan is not None:
                fetches["grad_has_inf_nan"] = self.grad_has_inf_nan
            if self.loss_has_inf_nan is not None:
                fetches["train_has_inf_nan"] = [
                    self.loss_has_inf_nan,
                    self.grad_has_inf_nan,
                ]

        return (enqueue_ops, fetches)

    def _check_grads(self, grads):
        has_inf_nan_list = []
        for (grad, _) in grads:
            has_inf_nan_list.append(
                tf.cast(
                    tf.reduce_sum(tf.cast(tf.is_inf(grad), dtype=tf.int32)),
                    dtype=tf.bool,
                )
            )
            has_inf_nan_list.append(
                tf.cast(
                    tf.reduce_sum(tf.cast(tf.is_nan(grad), dtype=tf.int32)),
                    dtype=tf.bool,
                )
            )
        self.grad_has_inf_nan = tf.reduce_all(has_inf_nan_list)
        assert self.grad_has_inf_nan is not None

    def _add_forward_pass_and_gradients(self, datas):
        with tf.device(self.cpu_device):
            gpu_copy_stage = data_flow_ops.StagingArea(
                self.key_types, shapes=self.get_data_list_shape(datas)
            )
            gpu_copy_stage_op = gpu_copy_stage.put(datas)
            datas = gpu_copy_stage.get()

        with tf.device(self.device):
            gpu_compute_stage = data_flow_ops.StagingArea(
                self.key_types, shapes=self.get_data_list_shape(datas)
            )
            gpu_compute_stage_op = gpu_compute_stage.put(datas)
            datas = gpu_compute_stage.get()

        with tf.device(self.device):
            loss, info_list = self.network.build_graph(datas[0], self.global_step)
            params = tf.trainable_variables()
            aggmeth = tf.AggregationMethod.DEFAULT

            if self.config_manager.use_mix_precision:
                self.opt = self.network.get_optimizer()
                self.opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
                    self.opt
                )
                gradvars = self.opt.compute_gradients(loss)
            else:
                grads = tf.gradients(loss, params, aggregation_method=aggmeth)
                gradvars = list(zip(grads, params))

            if self.config_manager.check_values:
                self.loss_has_inf_nan = tf.logical_or(tf.is_inf(loss), tf.is_nan(loss))
                assert self.loss_has_inf_nan is not None
                self._check_grads(gradvars)

            gradvars, max_noisescale = self.gradient_fusion.run(gradvars)

            return (
                loss,
                info_list,
                gradvars,
                max_noisescale,
                gpu_copy_stage_op,
                gpu_compute_stage_op,
            )
