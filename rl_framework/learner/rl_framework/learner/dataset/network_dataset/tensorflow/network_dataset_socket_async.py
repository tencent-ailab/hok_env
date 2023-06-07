import multiprocessing
import os
import sys


import numpy as np
import tensorflow as tf
from rl_framework.learner.dataset.lock_free_queue.lock_free_queue_shallow import (
    SharedCircBuf,
)
from rl_framework.learner.dataset.network_dataset import NetworkDatasetBase
from rl_framework.mem_pool import MemPoolAPIs, SamplingStrategy


class NetworkDataset(NetworkDatasetBase):
    def __init__(self, config_manager, AdapterClass):
        if config_manager.use_fp16:
            self.key_types = [tf.float16]
        else:
            self.key_types = [tf.float32]
        self.batch_size = config_manager.batch_size
        self.data_keys = config_manager.data_keys
        self.data_shapes = AdapterClass.get_data_shapes()
        self.init_index = config_manager.hvd_rank
        self.server_ips = config_manager.ips
        self.server_ports = config_manager.ports
        self.mem_process_num = config_manager.mem_process_num

        self.offline_rl_info_adapter = AdapterClass()
        example = self._get_sample_batch()
        self.cbuf = SharedCircBuf(16, example, self.data_keys)
        for i in range(self.mem_process_num):
            pid = multiprocessing.Process(target=self.enqueue_data, args=(i,))
            pid.daemon = True
            pid.start()

        self.enqueue_datas = []
        self.sample_num = 0

    def _get_dtype(self, dtype_str):
        if dtype_str == "float32":
            dtype = np.float32
        elif dtype_str == "float64":
            dtype = np.float64
        elif dtype_str == "int32":
            dtype = np.int32
        elif dtype_str == "int64":
            dtype = np.int64
        else:
            dtype = None
        return dtype

    def get_next_batch(self):
        return tf.py_func(self.next_batch, [], self.key_types)

    def next_batch(self):
        sdata_list = []
        get_data = 0
        while get_data == 0:
            get_data = self.cbuf.get_size()
        if self.cbuf.get_size() == 0:
            print(
                "%d get next_batch: queuesize %d" % (os.getpid(), self.cbuf.get_size())
            )
            sys.stdout.flush()
        tdata_list = self.cbuf.get()
        for i in range(len(tdata_list)):
            if self.key_types[i] == tf.float32:
                sdata_list.append(np.array(tdata_list[i]))
            elif self.key_types[i] == tf.int32:
                sdata_list.append((np.int32(tdata_list[i])))
        return sdata_list

    def _get_sample_batch(self):
        sample_batch = {}
        data_shapes_num = len(self.data_shapes)
        for i in range(data_shapes_num):
            dtype = self._get_dtype(self.key_types[i])
            if dtype in (np.float32, np.float64):
                data = np.random.rand(
                    *([self.batch_size] + self.data_shapes[i])
                ).astype(dtype)
            elif dtype in (np.int32, np.int64):
                data = np.random.randint(
                    5, size=([self.batch_size] + self.data_shapes[i])
                ).astype(dtype)
            else:
                pass
            sample_batch[self.data_keys[i]] = data
        return sample_batch

    def enqueue_data(self, i):
        sample_num = 0
        enqueue_datas = []
        for i in range(len(self.data_keys)):
            enqueue_datas.append([])
        index = self.init_index * self.mem_process_num + i
        index = index % (len(self.server_ports) * len(self.server_ips))
        server_ip = self.server_ips[0]
        server_port = self.server_ports[index % len(self.server_ports)]
        api = MemPoolAPIs(server_ip, server_port, "mcp")
        while True:
            _, sample = api.pull_sample(SamplingStrategy.PriorityGet.value)
            enqueue_data = self.offline_rl_info_adapter.deserialization(sample)
            for i in range(len(self.data_keys)):
                enqueue_datas[i].append(enqueue_data[i])
            sample_num += 1
            if sample_num >= self.batch_size:
                self.cbuf.put(enqueue_datas)
                for i in range(len(enqueue_datas)):
                    enqueue_datas[i] = []
                sample_num = 0
