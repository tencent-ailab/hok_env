#  definition of node and gradient fusion.
try:
    import horovod.tensorflow as hvd
    hvd.init()
    has_hvd = True
except Exception:  # pylint: disable=broad-except
    has_hvd = False
import collections

import numpy as np
import tensorflow as tf
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.common.logging import logger as LOG


#  The class describes some information about the node
class NodeInfo(object):
    #  The constructor. Initialization of node details
    def __init__(self):
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

    #  Provides tensorflow-op unified initialization method of training node parameters.
    #  When horovod is not installed in the environment, no operation is performed.
    def get_bcast_op(self):
        if has_hvd:
            return hvd.broadcast_global_variables(0)
        else:
            return tf.no_op()


class GradientFusion(object):
    #  The constructor
    def __init__(self):
        #  Singleton object of framework configuration
        self.config_manager = ConfigControl()
        #  A number of samples processed before the model is updated
        self.batch_size = self.config_manager.batch_size
        #  An object that stores information about a node
        node_info = NodeInfo()
        #  Total number of training nodes, and In general, it is the number
        #  of GPUs used in the experiment
        self.num_gpus = node_info.size

    #  This function reduces the gradient of all layers in the network, and it will call
    #  different gradient fusion implementations according to different configurations.
    #  @param gradvars It contains gradient tensors of all layers in the network
    def run(self, gradvars):
        if not has_hvd or self.num_gpus == 1:
            max_noisescale = self._calculate_noise(
                gradvars, gradvars, 1, self.batch_size
            )
            return gradvars, max_noisescale

        self.compression = hvd.Compression.none
        if self.config_manager.grad_to_fp16:
            self.compression = hvd.Compression.fp16

        if self.config_manager.sparse_as_dense:
            gradvars = [
                (tf.convert_to_tensor(value[0]), value[1])
                if value[0] is not None and isinstance(value[0], tf.IndexedSlices)
                else value
                for value in gradvars
            ]
        if self.config_manager.use_xla_fusion:
            gradvars, max_noisescale = self.xlascope_fusion(
                gradvars, self.num_gpus, self.batch_size
            )
        elif self.config_manager.use_fusion:
            gradvars, max_noisescale = self._piecewise_fusion(
                self.config_manager.piecewise_fusion_schedule,
                gradvars,
                self.num_gpus,
                self.batch_size,
                self.compression,
            )
        else:
            gradvars, max_noisescale = self._grad_fusion(
                gradvars, self.num_gpus, self.batch_size, self.compression
            )
        return gradvars, max_noisescale

    #  This function reduces the gradient of all layers in the network, which calls hvd.allreduce() one by
    #  one through each layer of the network to achieve reduction
    #  @param gradvars It contains gradient tensors of all layers in the network
    #  @param num_gpus Total number of training nodes
    #  @param batch_size A number of samples processed before the model is updated
    #  @param compression Compression algorithm used to reduce the amount of data sent
    #  and received by each worker node. Defaults to not using compression.
    def _grad_fusion(self, gradvars, num_gpus, batch_size, compression):
        grads_beforeavg = []
        params_new = []
        allgrads = []
        for (grad, param) in gradvars:
            if grad is not None:
                grads_beforeavg.append(grad)
                params_new.append(param)
                grad_hvd = hvd.allreduce(
                    grad, average=True, device_dense="", compression=compression
                )
                allgrads.append(grad_hvd)
        if self.config_manager.use_grad_clip:
            allgrads, _ = tf.clip_by_global_norm(
                allgrads, self.config_manager.grad_clip_range
            )
        gradvars = list(zip(allgrads, params_new))
        max_noisescale = self._calculate_noise(
            allgrads, grads_beforeavg, num_gpus, batch_size
        )
        return gradvars, max_noisescale

    #  This function reduces the gradient of all layers in the network. It adds an optimization point,
    #  which will merge the adjacent small gradients into large tensors for reduction
    #  @param piecewise_fusion_schedule Segmentation points for merging adjacent tensors. For example, when
    #  piecewise_fusion_schedule is equal to '3;8', gradvars[:3], gradvars[3:8] and gradvars[8:] will be
    #  merged into a large tensor, and then hvd.allreduce() will be called for gradient reduction
    #  @param gradvars It contains gradient tensors of all layers in the network
    #  @param num_gpus Total number of training nodes
    #  @param batch_size A number of samples processed before the model is updated
    #  @param compression Compression algorithm used to reduce the amount of data sent
    #  and received by each worker node. Defaults to not using compression.
    def _piecewise_fusion(
        self, piecewise_fusion_schedule, gradvars, num_gpus, batch_size, compression
    ):
        boundaries = []
        tensors_with_shapes = []
        allgrads = []
        grads_beforeavg = []
        params_new = []
        flat_tensors = {}
        orig_shapes = {}
        orig_sizes = {}
        pieces = piecewise_fusion_schedule.split(";")
        for piece in pieces:
            boundaries.append(int(piece))
        indexs = np.arange(len(boundaries) + 1)
        for i in range(len(boundaries) + 1):
            flat_tensors[i] = []
            orig_shapes[i] = []
            orig_sizes[i] = []

        for d, (grad, param) in enumerate(gradvars):
            if grad is not None:
                grad1 = grad
                grads_beforeavg.append(grad1)
                params_new.append(param)

                if d < boundaries[0]:
                    index = indexs[0]
                    flat_tensors[index].append(tf.reshape(grad1, [-1]))
                    orig_shapes[index].append(grad1.shape)
                    orig_sizes[index].append(grad1.shape.num_elements())
                if d >= boundaries[-1]:
                    index = indexs[-1]
                    flat_tensors[index].append(tf.reshape(grad1, [-1]))
                    orig_shapes[index].append(grad1.shape)
                    orig_sizes[index].append(grad1.shape.num_elements())

                for low, high, index in zip(
                    boundaries[:-1], boundaries[1:], indexs[1:-1]
                ):
                    if (d >= low) and (d < high):
                        flat_tensors[index].append(tf.reshape(grad1, [-1]))
                        orig_shapes[index].append(grad1.shape)
                        orig_sizes[index].append(grad1.shape.num_elements())

        for i in range(len(boundaries) + 1):
            if hvd.rank() == 0:
                LOG.info(orig_sizes[i])

            concatenated_grad = tf.concat(flat_tensors[i], 0)
            concatenated_grad_hvd = hvd.allreduce(
                concatenated_grad,
                average=True,
                device_dense="",
                compression=compression,
            )
            tensors_with_sizes = tf.split(concatenated_grad_hvd, orig_sizes[i])
            tensors_with_shapes.append(
                [
                    tf.reshape(grad, shape)
                    for grad, shape in zip(tensors_with_sizes, orig_shapes[i])
                ]
            )

        for i in range(len(boundaries) + 1):
            allgrads.extend(tensors_with_shapes[i])

        if self.config_manager.use_grad_clip:
            allgrads, _ = tf.clip_by_global_norm(
                allgrads, self.config_manager.grad_clip_range
            )

        gradvars = list(zip(allgrads, params_new))
        max_noisescale = self._calculate_noise(
            allgrads, grads_beforeavg, num_gpus, batch_size
        )
        return gradvars, max_noisescale

    def xlascope_fusion(self, gradvars, num_gpus, batch_size):
        tensors_with_shapes = []
        allgrads = []
        allparams = []
        grads_beforeavg = []
        params_new = {}
        flat_tensors = {}
        orig_shapes = {}
        orig_sizes = {}
        grad_scopes = collections.OrderedDict()
        gradvar_scopes = collections.OrderedDict()

        for i, (grad, param) in enumerate(gradvars):
            if grad is not None:
                grads_beforeavg.append(grad)
                if not (hasattr(grad, "shape")) or grad.shape.num_elements() is None:
                    allgrads.append(hvd.allreduce(grad, average=True, device_dense=""))
                    allparams.append(param)
                else:
                    grad_scope = []

                    def method_a():
                        xla_scope = grad.op.get_attr("_XlaScope").decode()
                        grad_scope.append(xla_scope)

                    def method_b():
                        xla_scope = param.op.get_attr("_XlaScope").decode()
                        grad_scope.append(xla_scope)

                    def method_c():
                        queue = collections.deque()
                        queue.append(param.op)
                        reached_ops = set()
                        while queue:
                            op = queue.pop()
                            if op not in reached_ops:
                                reached_ops.add(op)
                                for output in op.outputs:
                                    for consumer in output.consumers():
                                        try:
                                            xla_scope = consumer.get_attr(
                                                "_XlaScope"
                                            ).decode()
                                            grad_scope.append(xla_scope)
                                        except ValueError:
                                            queue.append(consumer)

                    for proc in [method_a, method_b, method_c]:
                        try:
                            proc()
                        except ValueError:
                            continue

                    if len(grad_scope) == 0:
                        LOG.info(
                            "%dth grad has no _XlaScope, failed to use xlascope_fusion."
                            % i
                        )
                        allgrads.append(
                            hvd.allreduce(grad, average=True, device_dense="")
                        )
                        allparams.append(param)
                    else:
                        grad_scope2 = list(set(grad_scope))
                        grad_scope2.sort()

                        grad_scopes[(grad, param)] = "_".join(grad_scope2)

        for k, v in grad_scopes.items():
            if v in gradvar_scopes:
                gradvar_scopes[v].append(k)
            else:
                gradvar_scopes[v] = [k]

        indexs = len(gradvar_scopes.keys())
        for i in range(indexs):
            flat_tensors[i] = []
            orig_shapes[i] = []
            orig_sizes[i] = []
            params_new[i] = []

        index = 0
        for _, value in gradvar_scopes.items():
            for grad, var in value:
                flat_tensors[index].append(tf.reshape(grad, [-1]))
                orig_shapes[index].append(grad.shape)
                orig_sizes[index].append(grad.shape.num_elements())
                params_new[index].append(var)
            index += 1

        for i in range(indexs):
            LOG.info("[rank: {}] fusion grad: {}", hvd.rank(), orig_sizes[i])
            if len(flat_tensors[i]) == 1:
                concatenated_grad = flat_tensors[i][0]
                concatenated_grad_hvd = hvd.allreduce(
                    concatenated_grad, average=True, device_dense=""
                )
                tensors_with_shapes.append(
                    [tf.reshape(concatenated_grad_hvd, orig_shapes[i][0])]
                )
            else:
                concatenated_grad = tf.concat(flat_tensors[i], 0)
                concatenated_grad_hvd = hvd.allreduce(
                    concatenated_grad, average=True, device_dense=""
                )
                tensors_with_sizes = tf.split(concatenated_grad_hvd, orig_sizes[i])
                tensors_with_shapes.append(
                    [
                        tf.reshape(grad, shape)
                        for grad, shape in zip(tensors_with_sizes, orig_shapes[i])
                    ]
                )

        for i in range(indexs):
            allgrads.extend(tensors_with_shapes[i])
            allparams.extend(params_new[i])

        if self.config_manager.use_grad_clip:
            allgrads, _ = tf.clip_by_global_norm(
                allgrads, self.config_manager.grad_clip_range
            )

        gradvars = list(zip(allgrads, allparams))
        max_noisescale = self._calculate_noise(
            allgrads, grads_beforeavg, num_gpus, batch_size
        )
        return gradvars, max_noisescale

    #  The gradient difference before and after fusion was calculated
    #  @param grads_afteravg Network gradient after fusion
    #  @param grads_beforeavg Network gradient before fusion
    #  @param num_gpus Total number of training nodes
    #  @param batch_size A number of samples processed before the model is updated
    def _calculate_noise(self, grads_afteravg, grads_beforeavg, num_gpus, batch_size):
        g_grad_norm_beforeavg = tf.global_norm(grads_beforeavg)
        g_grad_norm_beforeavg = g_grad_norm_beforeavg * g_grad_norm_beforeavg
        g_grad_norm_afteravg = tf.global_norm(grads_afteravg)
        g_grad_norm_afteravg = g_grad_norm_afteravg * g_grad_norm_afteravg

        if num_gpus == 1:
            batchsize = float(batch_size)
            noise_g = (
                batchsize * g_grad_norm_afteravg - batchsize * g_grad_norm_beforeavg + 1
            )
            noise_s = g_grad_norm_afteravg - g_grad_norm_beforeavg + 1
        else:
            large_batchsize = float(num_gpus * batch_size)
            small_batchsize = float(batch_size)
            alpha = 1 / (large_batchsize - small_batchsize)
            beta = 1 / (1 / small_batchsize - 1 / large_batchsize)
            noise_g = alpha * (
                large_batchsize * g_grad_norm_afteravg
                - small_batchsize * g_grad_norm_beforeavg
            )
            noise_s = beta * (g_grad_norm_beforeavg - g_grad_norm_afteravg)

        noise_scale = tf.div(noise_s, noise_g)
        relnoise_scale = noise_scale
        if num_gpus == 1:
            max_noisescale = relnoise_scale
        else:
            max_noisescale = hvd.allreduce(
                relnoise_scale, average=True, device_dense=""
            )
        return max_noisescale
