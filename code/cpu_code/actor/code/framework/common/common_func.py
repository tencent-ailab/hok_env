# -*- coding: utf-8 -*-
import configparser
import datetime
import os
import socket
import struct
import subprocess
import sys

import zmq
from framework.common.common_log import CommonLogger
from framework.common.common_log import g_log_time

LOG = CommonLogger.get_logger()


class ScrollConfig:
    def __init__(self, config_file_path):
        get_config = configparser.ConfigParser()
        get_config.read(config_file_path)
        sessions = get_config.sections()
        self.json_config = {}
        for session in sessions:
            self.json_config[session] = {}
            keys = get_config.options(session)
            for key in keys:
                self.json_config[session][key] = get_config.get(session, key)


def get_json_config():
    scroll_config = ScrollConfig("./config/game_ai_config.txt")
    return scroll_config.json_config


def log_time(text):
    def decorator(func):
        def wrapper(*args, **kws):
            start = datetime.datetime.now()
            result = func(*args, **kws)
            end = datetime.datetime.now()
            time = (end - start).seconds * 1000.0 + (end - start).microseconds / 1000.0
            g_log_time[text].append(time)
            return result

        return wrapper

    return decorator


def log_time_func(text, end=False):
    if g_log_time.get(text) is None:
        g_log_time[text] = []
    now = datetime.datetime.now()
    if len(g_log_time[text]) > 0:
        start = g_log_time[text][-1]
        if not isinstance(start, float):
            t = (now - start).seconds * 1000.0 + (now - start).microseconds / 1000.0
            g_log_time[text][-1] = t
    if not end:
        g_log_time[text].append(now)


"""
common_func class
"""


class CommonFunc:
    """
    machine local ip for reward info
    """

    @staticmethod
    def get_local_ip():
        cmd = "bash ../tool/get_ip.sh"
        local_ip = subprocess.check_output(cmd, shell=True)
        return local_ip.decode().strip()

    """
        game_id for a new game
    """

    @staticmethod
    def get_game_id():
        return (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
            + "_"
            + str(os.getpid())
        )

    """
        check point version from check point name
    """

    @staticmethod
    def get_version(path_name, key_word):
        return path_name.split("/")[-1].replace(key_word, "")

    """
        pack 128 samples to mempool
    """

    @staticmethod
    def generate_data(*samples):
        send_samples = []
        start_idx = 0
        end_idx = int(len(samples))
        batch_idx = 0

        while start_idx < end_idx:
            sample_num = min(end_idx - start_idx, 128)
            sys.stdout.flush()
            send_sample = (
                struct.pack("<I", int(batch_idx))
                + struct.pack("<I", int(1000001))
                + struct.pack("<I", int(sample_num))
            )
            str_com = ""
            for frame_idx in range(0, sample_num):
                frame_no = start_idx + frame_idx
                str_com = samples[frame_no]
                send_sample += struct.pack("<I", int(len(str_com))) + struct.pack(
                    "<%ss" % (len(str_com)), str_com
                )
            send_sample = (
                struct.pack("<I", socket.htonl(int(len(send_sample)) + 4)) + send_sample
            )
            send_samples.append(send_sample)
            start_idx = start_idx + sample_num
        return send_samples


"""
Initialize the socket connection between docker and GPU,
so that the gamecore can send data to memory buffer in GPU
"""


class Client:
    """
    create a new client for sending data to mempool
    """

    def __init__(self, ip, port, is_use_zmq=False):
        self.is_use_zmq = is_use_zmq
        try:
            if self.is_use_zmq:
                context = zmq.Context()
                self.socket_client = context.socket(zmq.REQ)
                addr = "tcp://{}:{}".format(ip, port)
                self.socket_client.connect(addr)
            else:
                self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket_client.connect((str(ip), int(port)))
        except socket.error as error:
            LOG.error("create socket error : %s" % error)
        LOG.info("Client Connect!")

    """
        send data to mempool
    """

    def send_data(self, data):
        try:
            if self.is_use_zmq:
                self.socket_client.send(data)
            else:
                self.socket_client.sendall(data)
            return True
        except socket.error as error:
            LOG.error("client send data error : %s" % error)
            return False
        except Exception as error:  # pylint: disable=broad-except
            LOG.error("client send data error : %s" % error)
            return False

    """
        recv data from mempool
    """

    def recv_data(self):
        try:
            if self.is_use_zmq:
                self.socket_client.recv()
            else:
                self.socket_client.recv(1024)
            return True
        except socket.error as error:
            LOG.error("client recv data error : %s" % error)
            return False
        except Exception as error:  # pylint: disable=broad-except
            LOG.error("client recv data error : %s" % error)
            return False

    """
        release client
    """

    def __del__(self):
        self.socket_client.close()
