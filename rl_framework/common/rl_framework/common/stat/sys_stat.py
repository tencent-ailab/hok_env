# -*- coding:utf-8 -*-

import psutil
import numpy as np


class SysStats:
    def __init__(self) -> None:
        self.network_sent_MB = None
        self.network_recv_MB = None
        self.last_network_sent = None
        self.last_network_recv = None

    @staticmethod
    def cpu_usage():
        cpu_usages = psutil.cpu_percent(percpu=True)
        return np.mean(cpu_usages), np.sum(cpu_usages)

    @staticmethod
    def cpu_count():
        return psutil.cpu_count()

    @staticmethod
    def total_memory_GB():
        mem = psutil.virtual_memory()
        return float(mem.total) / (1024 ** 3)

    @staticmethod
    def memory_usage_GB():
        mem = psutil.virtual_memory()
        return float(mem.used) / (1024 ** 3)

    @staticmethod
    def network_stats(duration_sec):
        network_sent = 0
        network_recv = 0

        curr_network_sent = int(
            psutil.net_io_counters()[0] / (1024 ** 2)
        )  # 上传的数据总量(MB)
        curr_network_recv = int(
            psutil.net_io_counters()[1] / (1024 ** 2)
        )  # 下载的数据总量(MB)

        if self.last_network_sent:
            self.network_sent = curr_network_sent - self.last_network_sent
            network_sent = self.network_sent / duration_sec

        if self.last_network_recv:
            self.network_recv = curr_network_recv - self.last_network_recv
            self.last_network_recv = curr_network_recv
            network_recv = self.network_recv / duration_sec

        self.last_network_sent = curr_network_sent
        self.last_network_recv = curr_network_recv

        return network_sent, network_recv
