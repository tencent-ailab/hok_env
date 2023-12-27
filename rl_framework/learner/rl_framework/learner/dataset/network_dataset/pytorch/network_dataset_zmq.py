# -*- coding:utf-8 -*-
import multiprocessing
import os

import lz4.block

from rl_framework.learner.dataset.network_dataset.common.batch_process import (
    BatchProcess,
)
from rl_framework.learner.dataset.network_dataset.common.sample_manager import MemBuffer
from rl_framework.mem_pool.zmq_mem_pool_server.zmq_mem_pool import ZMQMEMPOOL
from rl_framework.common.logging import logger as LOG


class NetworkDataset(object):
    def __init__(self, config_manager, adapter, port=35200):
        self.max_sample = config_manager.max_sample
        self.batch_size = config_manager.batch_size
        self.adapter = adapter
        self.data_shapes = self.adapter.get_data_shapes()
        self.use_fp16 = config_manager.use_fp16
        self.membuffer = MemBuffer(
            config_manager.max_sample, self.data_shapes[0][0], self.use_fp16
        )

        self.batch_process = BatchProcess(
            self.batch_size,
            self.data_shapes[0][0],
            config_manager.batch_process,
            self.use_fp16,
        )

        self.port = port
        self.zmq_mem_pool = ZMQMEMPOOL(self.port)
        self.init_dataset = False

        for i in range(config_manager.sample_process):
            pid = multiprocessing.Process(target=self.enqueue_data, args=(i,))
            pid.daemon = True
            pid.start()

        self.batch_process.process(self.membuffer.get_sample)
        self.last_batch_index = -1

    def get_next_batch(self):
        batch_index, sample_buf = self.batch_process.get_batch_data()
        if self.last_batch_index >= 0:
            self.batch_process.put_free_data(self.last_batch_index)
        self.last_batch_index = batch_index

        return sample_buf

    def enqueue_data(self, process_index):
        LOG.info(
            "sample process port:{} process_index:{} pid:{}".format(
                self.port, process_index, os.getpid()
            )
        )
        while True:
            for sample in self.zmq_mem_pool.pull_samples():
                decompress_data = lz4.block.decompress(
                    sample, uncompressed_size=3 * 1024 * 1024
                )
                sample_list = self.adapter.deserialization(decompress_data)
                for sample in sample_list:
                    self.membuffer.append(sample)

    def get_recv_speed(self):
        return self.membuffer.get_speed()
