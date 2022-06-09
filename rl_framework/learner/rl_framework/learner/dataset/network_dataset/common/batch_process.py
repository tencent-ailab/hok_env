import copy
import ctypes
import os
import time
from multiprocessing import Array, Process, Queue, Value

import numpy as np


class BatchManager(object):
    def __init__(self, batch_size, sample_size, process_num, use_fp16):
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.process_num = process_num
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.data_type = ctypes.c_uint16
        else:
            self.data_type = ctypes.c_float
        self.data = Array(
            self.data_type, process_num * 2 * sample_size * batch_size, lock=False
        )
        self.state = Array(ctypes.c_int, process_num * 2 + 1, lock=False)
        for index in range(len(self.state)):
            self.state[index] = 1
        self.last_get = Value("i", process_num * 2, lock=False)

    def set_batch_sample(self, sample, batch_index):
        if not (
            isinstance(sample, np.ndarray)
            or sample.shape == (1, self.sample_size * self.batch_size)
        ):
            print("set_batch_sample error batch", sample.shape)
            return False
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(
            self.process_num * 2, self.batch_size, self.sample_size
        )
        nparray[batch_index] = sample
        return True

    def set_one_sample(self, sample, batch_index, sample_index):
        if not (
            isinstance(sample, np.ndarray) or sample.shape == (1, self.sample_size)
        ):
            print("set_one_sample error sample", sample.shape)
            return False
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(
            self.process_num * 2 * self.batch_size, self.sample_size
        )
        nparray[batch_index * self.batch_size + sample_index] = sample
        return True

    def get_batch_sample(self, batch_index):
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(
            self.process_num * 2 * self.batch_size, self.sample_size
        )
        value = copy.deepcopy(
            nparray[
                batch_index * self.batch_size : batch_index * self.batch_size
                + self.batch_size
            ]
        )
        return value

    def set_state(self, index):
        self.state[self.last_get.value] = 1
        self.last_get.value = index

    def clear(self):
        for index in range(len(self.state)):
            self.state[index] = 1
        self.last_get.value = 2 * self.process_num


class BatchProcess(object):
    def __init__(self, batch_size, sample_size, process_num, use_fp16):
        self.batch_size = batch_size
        self.process_num = process_num
        self.use_fp16 = use_fp16
        self.batch_queue = Queue()
        self.batch_manager = BatchManager(
            batch_size=batch_size,
            sample_size=sample_size,
            process_num=process_num,
            use_fp16=self.use_fp16,
        )
        self.pids = []
        self.last_get_index = None

    def __process_run(self, process_index, get_sample_func, full_queue):
        print(
            "[BatchProcess::__process_run] process_index:{} pid:{}".format(
                process_index, os.getpid()
            )
        )
        while True:
            if self.batch_manager.state[2 * process_index] == 1:
                batch_index = 2 * process_index
            elif self.batch_manager.state[2 * process_index + 1] == 1:
                batch_index = 2 * process_index + 1
            else:
                time.sleep(0.005)
                continue
            for sample_index in range(self.batch_size):
                sample = get_sample_func()
                self.batch_manager.set_one_sample(sample, batch_index, sample_index)
            self.batch_manager.state[batch_index] = 0
            full_queue.put(batch_index)

    def process(self, get_data_func):
        for process_index in range(self.process_num):
            pid = Process(
                target=self.__process_run,
                args=(
                    process_index,
                    get_data_func,
                    self.batch_queue,
                ),
            )
            pid.daemon = True
            pid.start()
            self.pids.append(pid)

    def get_batch_data(self):
        if self.last_get_index is not None:
            self.batch_manager.set_state(self.last_get_index)
        batch_index = self.batch_queue.get()
        self.last_get_index = batch_index
        return self.batch_manager.get_batch_sample(batch_index)

    def exit(self):
        for pid in self.pids:
            pid.join()
