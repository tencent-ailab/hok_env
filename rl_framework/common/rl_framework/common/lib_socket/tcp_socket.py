# -*- coding:utf-8 -*-
import struct
import socket
import logging
import time


class TcpSocket:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = int(port)
        self.sock = None
        self._connect(self.ip, self.port)

    def _connect(self, ip, port):
        address = (ip, port)
        logging.info("address:%s" % str(address))
        while True:
            try:
                if self.sock:
                    self.sock.close()
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(address)
                return True
            except Exception as e:  # pylint: disable=broad-except
                logging.error("connect failed, address:%s, except:%s" % (address, e))
                time.sleep(1)

    def _send_all(self, request):
        try:
            _ = self.sock.sendall(request)
            return True
        except Exception as e:  # pylint: disable=broad-except
            logging.error("send failed, except:%s" % e)
            return False

    def _recv_all(self, recv_len):
        recved_len = 0
        recv_data = b""
        while recved_len < recv_len:
            try:
                data = self.sock.recv(recv_len - recved_len)
            except Exception as e:  # pylint: disable=broad-except
                logging.error("recv failed, except:%s" % e)
                return False, None
            if data == b"":
                logging.error("recv failed, data is empty")
                return False, None
            recv_data = recv_data + data
            recved_len += len(data)

        if recved_len != recv_len:
            logging.error("recv failed, recved_len != recv_len")
            return False, recv_data
        else:
            return True, recv_data

    def syn_send_recv(self, request):
        ret = True
        while True:
            # check status
            if not ret:
                logging.error("conn is error, try to reconnect")
                self._connect(self.ip, self.port)
                time.sleep(1)

            # send request
            ret = self._send_all(request)
            if not ret:
                logging.error("_send_all failed")
                continue

            # recv header
            head_length = 4
            ret, recv_data = self._recv_all(head_length)
            if not ret:
                logging.error("_recv_all data_len failed")
                continue

            # recv proto_data
            total_len = struct.unpack("I", recv_data)[0]
            total_len = socket.ntohl(total_len)
            if total_len - head_length > 0:
                ret, proto_data = self._recv_all(total_len - head_length)
                recv_data += proto_data
            if not ret:
                logging.error("_recv_all data failed")
                continue

            return recv_data
