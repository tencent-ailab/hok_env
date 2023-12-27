import multiprocessing
import struct
import sys
import time

import zmq

from rl_framework.mem_pool.mem_pool_api.mem_pool_protocol import CmdType
from rl_framework.common.logging import logger as LOG


class ZMQMEMPOOL(object):
    def __init__(self, port, max_message=2000):
        self.port = port
        self.data_queue = multiprocessing.Queue(max_message)
        self.recv_pid = multiprocessing.Process(
            target=self.recv_data, args=(self.data_queue,)
        )
        self.recv_pid.daemon = True
        self.recv_pid.start()

    def recv_data(self, queue):
        context = zmq.Context()
        recv_socket = context.socket(zmq.REP)
        addr = "tcp://*:{}".format(self.port)
        recv_socket.bind(addr)
        print_start_time = time.time()
        put_error_num = 0
        while True:
            if time.time() - print_start_time > 30:
                print_start_time = time.time()
                if put_error_num != 0:
                    LOG.info("queue put error: {}/min".format(put_error_num * 2))
                    put_error_num = 0

            data = recv_socket.recv()
            # data = recv_socket.recv(copy=False)
            recv_socket.send(b"success")
            try:
                has_deal = 8
                cmd_type = struct.unpack("I", data[has_deal : has_deal + 4])[0]
                if cmd_type == int(CmdType.KMemSetBatchRequest.value):
                    queue.put(self.generate_samples(data), block=False)
                elif cmd_type == int(CmdType.KMemCleanRequest.value):
                    queue.clear()
                else:
                    raise NotImplementedError
            except Exception:  # pylint: disable=broad-except
                msg = sys.exc_info()[0]
                print(msg)
                put_error_num += 1

    def generate_samples(self, data):
        sample_list = []
        has_deal = 12
        sample_num = struct.unpack("I", data[has_deal : has_deal + 4])[0]
        has_deal += 4
        for _ in range(sample_num):
            data_len = struct.unpack("I", data[has_deal : has_deal + 4])[0]
            has_deal += 4
            priority = struct.unpack("f", data[has_deal : has_deal + 4])[0]
            has_deal += 4
            sample_list.append(data[has_deal : has_deal + data_len])
            has_deal += data_len
        return sample_list

    def pull_samples(self):
        return self.data_queue.get()
