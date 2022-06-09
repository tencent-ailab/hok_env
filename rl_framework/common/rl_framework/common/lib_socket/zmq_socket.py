# -*- coding: utf-8 -*-
import logging
import zmq


class ZmqSocket:
    def __init__(self, ip_port, sock_type="client"):
        self.ip_port = ip_port
        self.timeout = 1000 * 30  # ms
        self.context = zmq.Context()
        self.socket = None
        self.poller_send = zmq.Poller()
        self.poller_recv = zmq.Poller()
        self._connect()

    def _connect(self):
        if self.socket:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
            self.poller_send.unregister(self.socket)
            self.poller_recv.unregister(self.socket)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.ip_port)
        self.poller_send.register(self.socket, zmq.POLLOUT)
        self.poller_recv.register(self.socket, zmq.POLLIN)

    def syn_send_recv(self, message):
        while True:
            if self.poller_send.poll(self.timeout):
                self.socket.send(message)
            else:
                logging.error("send timeout, try to reconnect")
                self._connect()
                continue

            if self.poller_recv.poll(self.timeout):
                data = self.socket.recv()
                break
            else:
                logging.error("recv timeout, try to reconnect")
                self._connect()
                continue
        return data

    def syn_recv_send(self, message):
        msg = self.syn_recv()
        self.syn_send(message)
        return msg

    def syn_recv(self):
        while True:
            socks = self.poller_recv.poll(self.timeout)
            # print(socks, type(socks))
            if socks:
                data = self.socket.recv()
                break
            else:
                logging.error("recv timeout, try to reconnect")
                self._connect()
        return data

    def syn_send(self, message):
        while True:
            socks = self.poller_send.poll(self.timeout)
            # print(socks, type(socks))
            if socks:
                self.socket.send(message)
                break
            else:
                logging.error("send timeout, try to reconnect")
                self._connect()
