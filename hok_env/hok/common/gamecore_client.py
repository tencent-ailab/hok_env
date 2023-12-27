import time
import requests

from hok.common.log import logger as LOG


class SimulatorType:
    Remote = "remote"  # 随机初始对战
    Repeat = "repeat"  # 优先从scene列表中回放对局到一定帧再对战, 当scene列表为空时, 采用remote模式
    RepeatOnly = "repeat_only"  # 从scene列表中回放对局到一定帧再对战, 当scene列表为空时, 会出错
    RemoteRepeat = "remote_repeat"  # 以一定概率(gamecore-server启动配置)选择remote或repeat模式, 当scene为空时, 采用remote模式


class GamecoreClient:
    def __init__(
        self,
        server_addr="127.0.0.1:23432",
        gamecore_req_timeout=30000,
        default_hero_config=None,
        max_frame_num=-1,
        simulator_type=SimulatorType.Remote,
    ) -> None:
        self.server_addr = server_addr
        self.default_hero_config = default_hero_config or {}
        self.gamecore_req_timeout = gamecore_req_timeout
        self.max_frame_num = max_frame_num
        self.simulator_type = simulator_type

    def start_game(
        self,
        runtime_id,
        servers,
        camp_hero_list,
        task_id=None,
        eval_mode=False,
        extra_abs_key_info=None,
    ):
        """
        当eval_mode为true时, 使用remote模式, 不使用repeat模式
        extra_abs_key_info:
            - 当保存对局的abs到scene中(用于回放)时, 附加额外的key信息
            - 当进行对局回放时, 根据key信息过滤筛选回放的abs
        """
        # camp_hero_list = {
        #     "mode": "1v1",
        #     "heroes": [
        #         [{"hero_id": 190, "skill_id": 80115, "symbol": [1512, 1512]}],
        #         [{"hero_id": 105, "skill_id": 80115, "symbol": [1512, 1512]}],
        #     ],
        # }
        start_config = {
            "hero_conf": [],
            "game_mode": camp_hero_list["mode"],
        }

        if self.max_frame_num > 0:
            start_config["max_frame_num"] = self.max_frame_num

        if extra_abs_key_info:
            start_config["extra_abs_key_info"] = extra_abs_key_info

        # 可选参数 task_id, 用于gamecore-server上报对局信息
        if task_id:
            start_config["task_id"] = task_id

        # 读取默认英雄信息, 生成最终对局配置
        for camp_id, hero_list in enumerate(camp_hero_list["heroes"]):
            for idx, hero_data in enumerate(hero_list):
                hero_id = hero_data["hero_id"]
                skill_id = hero_data.get(
                    "skill_id",
                    self.default_hero_config.get(hero_id, {}).get("skill_id"),
                )
                symbol = hero_data.get(
                    "symbol", self.default_hero_config.get(hero_id, {}).get("symbol")
                )

                hero_id = int(hero_id)
                if not servers[camp_id] or not all(servers[camp_id]):
                    # 当server为None, 或者是server的ip为"", 或者端口为0则作为common ai处理
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
                        "skill_id": skill_id,
                        "symbol": symbol,
                    }
                )

        simulator_type = self.simulator_type if not eval_mode else SimulatorType.Remote

        data = {
            "simulator_type": simulator_type,
            "runtime_id": runtime_id,
            "simulator_config": start_config,
        }
        self._send_http_request("newGame", data)

    def wait_game(self, runtime_id, max_timeout_second=0):
        start_time = time.time()
        while max_timeout_second <= 0 or time.time() - start_time <= max_timeout_second:
            try:
                if not self.check_exists_game(runtime_id):
                    break
            except Exception:
                LOG.exception("check_exists_battle failed")
            time.sleep(1)

    def check_exists_game(self, runtime_id):
        data = {
            "runtime_id": runtime_id,
        }
        ret = self._send_http_request("exists", data)
        if ret.get("exists"):
            return True
        return False

    def stop_game(self, runtime_id):
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
        resp.raise_for_status()
        if resp.ok:
            return resp.json() if json_ret else resp.content
        else:
            raise Exception(
                "Send request failed: {} {} {}".format(url, data, resp.content)
            )

    def task_list(self):
        """
        Get all task ids in gamecore server
        Return list of task id
        """
        ret = self._send_http_request("taskList", {})
        return ret["data"].get("task_ids") or []

    def task_detail(self, task_id):
        """
        Get all gamestate detail of taks_id
        return list of gamestate
        """
        data = {
            "task_id": task_id,
        }
        ret = self._send_http_request("taskDetail", data)
        return ret["data"].get("game_states") or []

    def task_remove(self, task_id):
        """
        Remove all gamestate of task_id
        """
        data = {
            "task_id": task_id,
        }
        self._send_http_request("taskRemove", data)

    def task_clear(self):
        """
        Clear all gamestate of all task
        """
        self._send_http_request("taskClear", {})
