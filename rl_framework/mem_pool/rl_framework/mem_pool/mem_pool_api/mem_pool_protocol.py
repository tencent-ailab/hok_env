import struct
import socket
from enum import Enum
import lz4.block

from rl_framework.common.logging import logger as LOG


class SamplingStrategy(Enum):
    MIN = 0
    RandomGet = 0
    PriRandomGet = 1
    PriorityGet = 2
    FIFOGet = 3
    LIFOGet = 4
    MAX = 4


class CmdType(Enum):
    KMemSetBatchRequest = 1000001
    KMemGetRequest = 2000000
    KMemGetBatchRequest = 2000001
    KMemGetBatchCompressRequest = 2000002
    KMemCleanRequest = 3000000


class MemPoolProtocol:
    def __init__(self):
        pass

    def format_get_request(self, search_id=10000, strategy=0):
        KMemGetRequest = int(CmdType.KMemGetRequest.value)
        head_length = 4 + 4 + 4 + 4  # total, seq, cmd, strategy

        return (
            struct.pack("<I", socket.ntohl(head_length))
            + struct.pack("<I", search_id)
            + struct.pack("<I", KMemGetRequest)
            + struct.pack("<I", strategy)
        )

    def parse_get_response(self, data):
        total = 0
        seq = 0
        cmd = 0
        sample = b""
        try:
            if len(data) >= 12:
                header = struct.unpack("III", data[0:12])
                total = socket.ntohl(header[0])
                seq = header[1]
                cmd = header[2]
                if len(data) > 12:
                    sample = data[12:]
        except Exception:  # pylint: disable=broad-except
            LOG.exception("parse data error")

        return total, seq, cmd, sample

    def format_set_batch_request(self, samples, priorities=None):
        if priorities is None:
            priorities = list([0.0] * len(samples))

        KMemSetBatchRequest = int(CmdType.KMemSetBatchRequest.value)

        # 1.compress each sample
        samples = self._compress_sample(samples)

        # 2.package samples
        sample_str = b""
        for frame_idx in range(0, len(samples)):
            sample = samples[frame_idx]
            sample_len = len(sample)
            priority = priorities[frame_idx]
            sample_str += (
                struct.pack("<I", int(sample_len))
                + struct.pack("<f", float(priority))
                + struct.pack("<%ss" % (sample_len), sample)
            )

        # 3.header info
        # total, seq, cmd, num, data
        total_len = 4 + 4 + 4 + 4 + int(len(sample_str))
        seq_no = 0
        sample_num = len(samples)
        # print ("sample num %s sample_str %s total_len %s" %(sample_num, len(sample_str), total_len))

        return (
            struct.pack("<I", socket.htonl(total_len))
            + struct.pack("<I", int(seq_no))
            + struct.pack("<I", int(KMemSetBatchRequest))
            + struct.pack("<I", int(sample_num))
            + sample_str
        )

    def parse_set_batch_response(self, data):
        total = 0
        seq = 0
        sample = b""
        try:
            if len(data) >= 8:
                header = struct.unpack("II", data[0:8])
                total = socket.ntohl(header[0])
                seq = header[1]
                if len(data) > 8:
                    sample = data[8:]
        except Exception:  # pylint: disable=broad-except
            LOG.exception("parse data error")

        return total, seq, sample

    def format_batch_samples_array(self, samples, priorities=None, max_sample_num=128):
        if priorities is None:
            priorities = list([0.0] * len(samples))

        start_idx = 0
        send_samples = []
        while start_idx < len(samples):
            sample_num = min(len(samples) - start_idx, max_sample_num)
            send_sample = self.format_set_batch_request(
                samples[start_idx : start_idx + sample_num],
                priorities[start_idx : start_idx + sample_num],
            )
            send_samples.append(send_sample)
            start_idx = start_idx + sample_num
        return send_samples

    def _compress_sample(self, samples):
        compress_samples = []
        for sample in samples:
            if isinstance(sample, str):
                sample = bytes(sample, encoding="utf8")
            if not isinstance(sample, bytes):
                return None

            compress_sample = lz4.block.compress(sample, store_size=False)
            compress_samples.append(compress_sample)
        return compress_samples

    def format_clean_request(self, search_id=10000):
        KMemGetRequest = int(CmdType.KMemCleanRequest.value)
        head_length = 4 + 4 + 4  # total, seq, cmd

        return (
            struct.pack("<I", socket.ntohl(head_length))
            + struct.pack("<I", search_id)
            + struct.pack("<I", KMemGetRequest)
        )
