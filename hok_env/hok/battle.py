import os
import socket
import time
import requests
import subprocess
import logging
import json

from hok.camp import camp_iterator


LOG = logging.getLogger(__file__)


default_hero_config_file = os.path.join(
    os.path.dirname(__file__), "default_hero_config.json"
)


def get_default_hero_config():
    default_hero_config = {}
    with open(default_hero_config_file) as f:
        data = json.load(f)
        for _hero_config in data:
            default_hero_config[_hero_config["hero_id"]] = _hero_config
    return default_hero_config


class Battle:
    def __init__(
        self, server_addr="127.0.0.1:23432", gamecore_req_timeout=30000
    ) -> None:
        self.server_addr = server_addr
        self.camp_iter = camp_iterator()
        self.default_hero_config = get_default_hero_config()
        self.gamecore_req_timeout = gamecore_req_timeout

    def start_server(
        self,
        server,
        server_path,
        server_port,
        server_log_path,
        server_driver,
    ):
        if server_driver == "local_tar":
            os.makedirs(server_path, exist_ok=True)
            cmd = "tar -C {} -xf {}".format(server_path, server)
            LOG.info(cmd)
            subprocess.run(cmd, env=os.environ, shell=True, check=True)
        elif server_driver == "url":
            os.makedirs(server_path, exist_ok=True)
            dst_file = os.path.join(server_path, "server.tgz")
            cmd = ["wget", "-O", dst_file, server]

            LOG.info(cmd)
            subprocess.run(cmd, env=os.environ, check=True)
            cmd = "tar -C {} -xf {}".format(server_path, dst_file)
            LOG.info(cmd)
            subprocess.run(cmd, shell=True, env=os.environ, check=True)
        elif server_driver == "server":
            # 已经启动的server, 直接返回
            return (server, server_port)
        elif server_driver == "common_ai":
            return None
        else:
            # server_driver == "local_dir"
            server_path = server
            pass

        os.makedirs(os.path.dirname(server_log_path), exist_ok=True)
        env = os.environ.copy()
        env["AI_SERVER_ADDR"] = "tcp://0.0.0.0:{}".format(server_port)
        cmd = "cd {}/code/cpu_code/script/; nohup sh start_server.sh >> {} 2>&1 &".format(
            server_path, server_log_path
        )
        LOG.info(cmd)
        subprocess.run(cmd, env=env, shell=True, check=True)
        return ("127.0.0.1", server_port)

    def _test_server(self, host, port):
        with socket.socket(socket.AF_INET) as s:
            try:
                s.connect((host, port))
            except ConnectionRefusedError:
                return False
        return True

    def wait_server(self, servers, timeout):

        end_time = time.time() + timeout
        all_done = False
        while time.time() < end_time and not all_done:
            try:
                all_done = True
                for server in servers:
                    if not server:
                        continue

                    if not self._test_server(server[0], server[1]):
                        all_done = False
                        break
            except Exception:
                LOG.exception("test server failed, continue")
            finally:
                time.sleep(1)

    def start_battle(self, runtime_id, servers):
        start_config = {
            "hero_conf": [],
        }
        for camp_id, hero_list in enumerate(next(self.camp_iter)):
            for idx, hero_id in enumerate(hero_list):
                hero_id = int(hero_id)
                if not servers[camp_id]:
                    request_info = {}
                elif idx == 0:
                    request_info = {
                        "ip": servers[camp_id][0],
                        "port": servers[camp_id][1],
                        "timeout": self.gamecore_req_timeout,
                    }
                else:
                    request_id = camp_id * len(hero_list)
                    request_info = {
                        "request_id": request_id,
                    }
                start_config["hero_conf"].append(
                    {
                        "hero_id": hero_id,
                        "request_info": request_info,
                        "skill_id": self.default_hero_config.get(hero_id, {}).get(
                            "skill_id"
                        ),
                        "symbol": self.default_hero_config.get(hero_id, {}).get(
                            "symbol"
                        ),
                    }
                )
        data = {
            "simulator_type": "remote",
            "runtime_id": runtime_id,
            "simulator_config": start_config,
        }
        self._send_http_request("newGame", data)

    def wait_battle(self, runtime_id):
        while True:
            try:
                if not self.check_exists_battle(runtime_id):
                    break
            except Exception:
                LOG.exception("check_exists_battle failed")
            time.sleep(1)

    def check_exists_battle(self, runtime_id):
        data = {
            "runtime_id": runtime_id,
        }
        ret = self._send_http_request("exists", data)
        if ret.get("exists"):
            return True
        return False

    def stop_battle(self, runtime_id):
        data = {
            "runtime_id": runtime_id,
        }
        self._send_http_request("stopGame", data, json_ret=False)

    def _send_http_request(self, req_type, data, json_ret=True):
        url = "http://%s/v2/%s" % (self.server_addr, req_type)
        headers = {
            "Content-Type": "application/json",
        }

        resp = requests.post(url=url, json=data, headers=headers, verify=False)
        if resp.ok:
            return resp.json() if json_ret else resp.content
        else:
            raise Exception(
                "Send request failed: {} {} {}".format(url, data, resp.content)
            )
