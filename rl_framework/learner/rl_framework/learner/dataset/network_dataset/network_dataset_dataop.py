import ctypes

import numpy as np
from rl_framework.learner.dataset.network_dataset import NetworkDatasetBase
from rl_framework.learner.dataset.network_dataset.common import *
from rl_framework.learner.dataset.network_dataset.common.shm_data_op import ShmDataOp


class NetworkDataset(NetworkDatasetBase):
    def __init__(self, config_manager, AdapterClass):
        self.task_id = 0
        self.str_task_uuid = "0"
        self.batch_size = config_manager.batch_size
        self.data_shapes = AdapterClass.get_data_shapes()
        self.init_index = config_manager.hvd_rank
        self.local_size = config_manager.hvd_local_size
        self.local_rank = config_manager.hvd_local_rank
        self.server_ports = config_manager.ports
        self.mem_process_num = config_manager.mem_process_num
        self.use_fp16 = config_manager.use_fp16

        max_index_block_size = ctypes.c_ulong(
            get_mem_pool_param(
                "max_index_block_size", self.server_ports, config_manager.mempool_path
            )
        )
        max_data_size = ctypes.c_ulong(
            get_mem_pool_param(
                "max_data_size", self.server_ports, config_manager.mempool_path
            )
        )
        self.sample_length = self.data_shapes[0][0]

        mem_pool_keys = get_mem_pool_key(self.server_ports, config_manager.mempool_path)
        print("dataop ", mem_pool_keys)

        each_key_num = int(len(mem_pool_keys) / self.local_size)
        temp_mempool_keys = np.arange(each_key_num, dtype=np.int32)
        for index in range(each_key_num):
            temp_mempool_keys[index] = mem_pool_keys[
                each_key_num * self.local_rank + index
            ]
        print("hvd local_rank:", self.local_rank, temp_mempool_keys)
        self.m_shm_dataop_op = ShmDataOp(
            self.mem_process_num,
            int(self.sample_length),
            int(self.batch_size),
            self.task_id,
            self.str_task_uuid,
            max_index_block_size,
            max_data_size,
            self.init_index,
            temp_mempool_keys,
            len(temp_mempool_keys),
            self.use_fp16,
        )

    def get_next_batch(self):
        consumer_datas = self.m_shm_dataop_op.comsumer_data()
        return [consumer_datas]
