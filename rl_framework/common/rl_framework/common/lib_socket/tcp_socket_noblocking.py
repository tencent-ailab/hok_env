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
                self.sock.setblocking(False)
                self.sock.connect(address)
                return True
            except BlockingIOError as e:
                return True
            except Exception as e:  # pylint: disable=broad-except
                logging.error("connect failed, address:%s, except:%s" % (address, e))
            time.sleep(1)

    def _check_conn_status(self):
        while True:
            try:
                data = self.sock.recv(1024)
                if data == b"":
                    logging.error("check conn is closed, try to reconnect")
                    self._connect(self.ip, self.port)
                else:
                    return True
            except BlockingIOError as e:
                return True
            except Exception as e:  # pylint: disable=broad-except
                logging.error("check conn except:%s" % e)
                self._connect(self.ip, self.port)
                # return False
            time.sleep(1)
        return True

    def _send_all(self, request):
        while True:
            try:
                _ = self.sock.sendall(request)
                return True
            except BlockingIOError:
                continue
            except socket.error as e:  # pylint: disable=broad-except
                logging.error("send failed, except:%s" % e)
                return False
        return True

    def _recv_all(self, recv_len):
        recved_len = 0
        recv_data = b""
        while recved_len < recv_len:
            try:
                data = self.sock.recv(recv_len - recved_len)
            except BlockingIOError:
                continue
            except Exception:  # pylint: disable=broad-except
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
        while True:
            # check conn status
            ret = self._check_conn_status()
            if not ret:
                logging.error("_check_conn_status failed")
                time.sleep(1)
                continue

            # send request
            ret = self._send_all(request)
            if not ret:
                logging.error("_send_all failed")
                time.sleep(1)
                continue

            # recv header
            head_length = 4
            ret, recv_data = self._recv_all(head_length)
            if not ret:
                logging.error("_recv_all data_len failed")
                time.sleep(1)
                continue

            # recv proto_data
            total_len = struct.unpack("I", recv_data)[0]
            total_len = socket.ntohl(total_len)
            if total_len - head_length > 0:
                ret, proto_data = self._recv_all(total_len - head_length)
                recv_data += proto_data
            if not ret:
                logging.error("_recv_all data failed")
                time.sleep(1)
                continue

            return recv_data
