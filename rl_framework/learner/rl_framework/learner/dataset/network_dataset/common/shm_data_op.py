from __future__ import absolute_import, division, print_function

import os
from ctypes import *

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import load_library


def _load_library(name, op_list=None):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.
      op_list: A list of names of operators that the library should have. If None
          then the .so file's contents will not be verified.

    Raises:
      NameError if one of the required ops is missing.
      NotFoundError if were not able to load .so file.
    """
    # filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(name)
    for expected_op in op_list or []:
        for lib_op in library.OP_LIST.op:
            if lib_op.name == expected_op:
                break
        else:
            raise NameError(
                "Could not find operator %s in dynamic library %s" % (expected_op, name)
            )
    return library


mem_pool_lib_path = (
    "/data1/reinforcement_platform/rl_learner_platform/code/shm_lib/mem_pool_dataop.so"
)
if "dataop" in os.environ.keys():
    mem_pool_lib_path = "{}/mem_pool_dataop.so".format(os.environ["dataop"])
MPI_LIB = _load_library(mem_pool_lib_path, ["ConsumerData"])
MPI_LIB_CTYPES = cdll.LoadLibrary(mem_pool_lib_path)


class ShmDataOp(object):
    object_count = 0

    def __init__(
        self,
        thread_num,
        sample_len,
        batch_size,
        task_id,
        task_uuid,
        max_data_size,
        hvd_rank,
        mem_pool_keys,
        mem_pool_key_size,
        is_fp16,
    ):
        self.m_index = ShmDataOp.object_count
        ShmDataOp.object_count += 1
        self.batch_size = batch_size
        self.sample_length = sample_len
        buff_length = self.sample_length * self.batch_size * thread_num * 2
        if is_fp16:
            self.data_type = tf.float16
            self.data_buff = np.arange(buff_length, dtype=np.float16)
        else:
            self.data_type = tf.float32
            self.data_buff = np.arange(buff_length, dtype=np.float32)
        str_task_uuid = task_uuid.encode("utf-8")
        INT_POINTER = POINTER(c_int)
        FLOAT_POINTER = POINTER(c_float)
        MPI_LIB_CTYPES.shmdata_init(
            int(thread_num),
            self.data_buff.ctypes.data_as(FLOAT_POINTER),
            mem_pool_keys.ctypes.data_as(INT_POINTER),
            int(mem_pool_key_size),
            sample_len,
            max_index_block_size,
            max_data_size,
            batch_size,
            task_id,
            str_task_uuid,
            hvd_rank,
        )

    def restart(self):
        MPI_LIB_CTYPES.restart(int(self.m_index))

    def set_mask(self, task_model_key):
        MPI_LIB_CTYPES.setMask(bytes(task_model_key, "utf-8"), int(self.m_index))

    def comsumer_data(self):
        initializer = tf.constant_initializer(10000.0)
        temp_datas = tf.get_variable(
            "input_datas_{}".format(self.m_index),
            shape=[2],
            initializer=initializer,
            trainable=False,
            dtype=self.data_type,
        )
        consumer_datas = MPI_LIB.consumer_data(
            temp_datas, self.sample_length * self.batch_size, int(self.m_index)
        )
        consumer_datas = tf.reshape(
            consumer_datas, [self.batch_size, self.sample_length]
        )
        return consumer_datas
