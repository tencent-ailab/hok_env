import json
import os
import random
import socket
import struct
import subprocess
import sys

from model_syn_base import ModelSynBase

sys.path.append("./mcp_opporation_tools/")


class ModelSynP2P(ModelSynBase):
    def __init__(self, address):
        ip, port = address.split(":")
        self.address = (ip.strip(), int(port.strip()))
        self.time_out = 10
        self.docker_model_path = "/code/cpu_code/actor/model/update/"

    def syn_model(self, model_path):
        base_path = os.path.abspath("..")
        src = []
        dst = []
        model_file = model_path
        final_use_done = model_file + ".done"
        model_name = model_file.split("/")[-1]
        src.append(model_file)
        src.append(final_use_done)
        dst.append("%s/%s" % (self.docker_model_path, model_name))
        dst.append("%s/%s.done" % (self.docker_model_path, model_name))

        os.system(
            'touch %s; echo "hello world" > %s' % (final_use_done, final_use_done)
        )
        ip_ports = []
        init_ip = ""
        with open(base_path + "/mcp_opporation_tools/current.iplist", "r") as fin:
            ip_ports = fin.readlines()
            random.shuffle(ip_ports)
            host_map = []
            succ_machine = 0
        print("before for loop")
        for ip_port in ip_ports:
            local_ip, port = ip_port.split()
            real_port = (int(port) - 36000) * 100 + 35297
            if succ_machine == 0:
                ret, _ = self._first_trans(base_path, local_ip, real_port, src, dst)
                if ret == 0:
                    init_ip = "%s %s" % (local_ip, real_port)
                    succ_machine += 1
                else:
                    host_map.append("%s %s" % (local_ip, real_port))
            else:
                host_map.append("%s %s" % (local_ip, real_port))
        if init_ip != "":
            self._p2p(host_map, init_ip, dst)
        else:
            print("error:first trans error")

    def _info_master(self, json_data):
        data = json.dumps(json_data).encode("utf=8")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1200)
        total_length = len(data) + 8
        search_id = random.randint(0, 50000)
        message = (
            struct.pack("<I", socket.ntohl(total_length))
            + struct.pack("<I", search_id)
            + data
        )
        try:
            client.connect(self.address)
            client.send(message)
            rsps = b""
            while True:
                rsp = client.recv(2048)
                if len(rsp) == 0:
                    break
                rsps += rsp
            rsps_json = json.loads(rsps[8:])
            return 0, rsps_json
        except Exception as error:  # pylint: disable=broad-except
            print("info_master exception:", error)
            return -1, traceback.format_exc()
        finally:
            client.close()

    def _p2p(self, host_map, init_ip, dsts):
        sync_files = ""
        for dst in dsts:
            sync_files += ",{}".format(dst.strip())
        json_data = {
            "Data_Type": "add_req",
            "Name": sync_files,
            "Ip": init_ip,
            "HostMap": host_map,
            "TaskId": -999,
        }
        ret, msg = self._info_master(json_data)
        if ret != 0:
            print("p2p error {}".format(msg))

    def _first_trans(self, base_path, local_ip, port, srcs, dsts):
        ret = 0
        for src, dst in zip(srcs, dsts):
            cmd = "%s/mcp_opporation_tools/p2p_tools %s %s %s %s %s" % (
                base_path,
                local_ip,
                port,
                self.time_out * 1000,
                src,
                dst,
            )
            ret, msg = subprocess.getstatusoutput(cmd)
            if ret != 0:
                return ret, msg
        return ret, "succ"
