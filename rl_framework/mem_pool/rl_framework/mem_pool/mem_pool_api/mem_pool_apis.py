#  @package mem_pool
#  Provides a mem pool api class to push samples to mempool and pull sample from mempool

from rl_framework.common.lib_socket.zmq_socket import ZmqSocket
from rl_framework.common.lib_socket.tcp_socket import TcpSocket
from rl_framework.mem_pool.mem_pool_api.mem_pool_protocol import MemPoolProtocol


class MemPoolAPIs(object):
    #  The constructor.
    #  @param self The object pointer.
    #  @param ip mempool server ip
    #  @param port mempool server port
    #  @param socket_type mempool server type,
    #         "zmq": mempool is a python version, use zeromq protocol
    #         "mcp++": mempool is a mcp++ version, use tcp protocol
    def __init__(self, ip, port, socket_type="zmq"):
        if socket_type == "zmq":
            ip_port = "tcp://%s:%s" % (ip, port)
            self._client = ZmqSocket(ip_port, "client")
        elif socket_type == "mcp++":
            self._client = TcpSocket(ip, port)
        else:
            raise NotImplementedError

        self.protocol = MemPoolProtocol()

    #  Pull sample Interface: randomly pull a sample from mempool
    #  @param self The object pointer.
    #  @param strategy sampling strategy type:int.
    #  @return seq sequence number
    #  @return sample
    def pull_sample(self, strategy):
        request = self.protocol.format_get_request(strategy=strategy)
        response = self._request(request)
        _, seq, _, sample = self.protocol.parse_get_response(response)
        return seq, sample

    #  Push samples Interface:
    #       compress each sample by lz4 and send to mempool
    #       if more than max_sample_num, split to packages, one package include max_sample_num samples
    #  @param self The object pointer.
    #  @param samples samples type:list, sample type:str or bytes
    #  @param priorities priorities type:list, priority type:float
    #  @param max_sample_num max_sample_num type:int, default 128
    #  @return ret_array, success or fail
    def push_samples(self, samples, priorities=None, max_sample_num=128):
        format_samples = self.protocol.format_batch_samples_array(
            samples, priorities, max_sample_num
        )
        ret_array = []
        for format_sample in format_samples:
            ret = self._request(format_sample)
            ret_array.append(ret)
        return ret_array

    def clean_samples(self):
        request = self.protocol.format_clean_request()
        response = self._request(request)
        return response

    def _request(self, data):
        return self._client.syn_send_recv(data)
