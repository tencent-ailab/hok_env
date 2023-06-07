# -*- coding: utf-8 -*-
import os
import traceback
import json
import time
from enum import Enum

import requests
import numpy as np


from rl_framework.common.logging import log_time
import rl_framework.common.logging as LOG
from camp import camp_iterator
from hok.hok3v3 import CONFIG_DAT

import hok.hok3v3.lib.lib3v3 as interface

default_hero_config_file = os.path.join(
    os.path.dirname(__file__), "default_hero_config.json"
)


class ResponseType(Enum):
    NONE = 0
    EMPTY = 1
    CACHED = 2
    GAMEOVER = 3


def get_default_hero_config():
    default_hero_config = {}
    with open(default_hero_config_file) as f:
        data = json.load(f)
        for _hero_config in data:
            default_hero_config[_hero_config["hero_id"]] = _hero_config
    return default_hero_config


class GameCoreClient:
    def __init__(self, host, seed, gc_server=None):
        self.camp_iter = camp_iterator()
        self.default_hero_config = get_default_hero_config()
        self.seed = seed
        self.host = host
        self.user_token = self.host.replace(".", "D")
        self.runtime_id = "{}-{}".format(self.user_token, self.seed)

        self.launch_server = "127.0.0.1:23432"
        if gc_server is not None:
            self.launch_server = gc_server
        self.num_player = 2
        # self.conns = [None] * self.num_player
        self.addrs = [None] * self.num_player
        self.ports = [None] * self.num_player
        self.player_ids = {}
        self.cur_sgame_ids = [None] * self.num_player

        self.lib_processor = interface.Interface()
        self.lib_processor.Init(CONFIG_DAT)
        self.is_gameover = True
        self.gameover_sent = True

        # ms of gamecore waiting for server reply
        # 除了特征处理本身耗时, 还需要考虑样本发送耗时, 因此需要设置长一些
        self.gamecore_req_timeout = 300000
        # ms of server waiting for game requests
        self.server_recv_timeout = 30000  # TODO hongyangqin
        self.num_retry = 5

        self.hero_num = 3

        self.player_masks = {}  # TODO hongyangqin, 去掉

        self.cur_frame_no = -1

        # when debug memory leak...
        self.fake_connect = False
        self.fake_connect_when = 3000
        # self.fake_recv = None
        self.fake_buffer = {}

    def _split_legal_action(self, legal_action, action_split_size):
        return np.split(legal_action, action_split_size)

    def step_action(self, probs, features, req_pb, agent_id, lstm_info):
        process_result = self._step_action(probs, agent_id)
        sample = self._sample_process(
            probs, features, process_result, req_pb, lstm_info
        )
        return sample

    def _gameover(self, agent_id):
        self._send(agent_id, ResponseType.GAMEOVER, -1, self.cur_sgame_ids[agent_id])

    def _sample_process(self, probs, features, process_result, req_pb, lstm_info):
        lstm_cell, lstm_hidden = lstm_info

        # probs : two camp , 3 heros, label prob +  value
        frame_no = req_pb.frame_no

        feature_one_camp = []
        reward_one_camp = []
        actions_one_camp = []
        sub_actions_one_camp = []
        legal_action_one_camp = []
        prob_one_camp = []
        value_one_camp = []
        is_train_one_camp = []
        done_one_camp = []
        all_hero_reward_one_camp = []
        # is_train_one_camp = []
        for hero in range(len(features)):
            feature_one_camp.append(np.array(features[hero].feature))
            legal_action_one_camp.append(sum(process_result[hero].legal_action, []))
            actions_one_camp.append(process_result[hero].actions)
            reward_one_camp.append(features[hero].reward)
            tmp_prob = sum(process_result[hero].final_prob_list, [])
            prob_one_camp.append(tmp_prob[:-1])
            value_one_camp.append(tmp_prob[-1])
            sub_actions_one_camp.append(process_result[hero].sub_actions)
            # hero_rid = state_dict_list[hero]['player_id']
            is_train_one_camp.append(process_result[hero].is_train)
            all_hero_reward_one_camp.append(features[hero].all_hero_reward_info)

            done_one_camp.append(False)

        keys = (
            "frame_no",
            "vec_feature_s",
            "legal_action_s",
            "action_s",
            "reward_s",
            "value_s",
            "prob_s",
            "sub_action_s",
            "lstm_cell",
            "lstm_hidden",
            "done",
            "is_train",
            "all_hero_reward_s",
        )
        values = (
            frame_no,
            feature_one_camp,
            legal_action_one_camp,
            actions_one_camp,
            reward_one_camp,
            value_one_camp,
            prob_one_camp,
            sub_actions_one_camp,
            lstm_cell,
            lstm_hidden,
            done_one_camp,
            is_train_one_camp,
            all_hero_reward_one_camp,
        )
        sample = dict(zip(keys, values))
        self.last_sample = sample
        return sample

    def update_skip(self, key, value):
        if not self.fake_connect:
            return
        self.fake_buffer[key] = value

    def _format_actions(self, actions):
        rp_actions = []

        # format shape
        data_shape = [13, 25, 42, 42, 39, 1]
        data_len = sum(data_shape)
        split_idx = []
        tmp_si = 0
        for s_i in range(len(data_shape)):
            tmp_si += data_shape[s_i]
            split_idx.append(tmp_si)
        # [:-1], skip last none list
        format_shape = split_idx[:-1]

        for i, camp in enumerate(actions):
            each_camp_heros_action = []
            for hero, action in enumerate(camp):
                # formulation check
                if isinstance(action, (tuple, list)):
                    if not len(action) == data_len:
                        LOG.error(
                            "action[{}] length incorrect: {}, but expect 6.".format(
                                hero, len(action)
                            )
                        )
                        assert False
                    action_raw = np.array(action)
                    action = np.split(action_raw, format_shape, axis=1)
                elif isinstance(action, np.ndarray):
                    # if not (len(action.shape) == 1 and action.shape[0] == 6):
                    if not (action.shape[1] == data_len):
                        LOG.error(
                            "action[{}] shape incorrect: {}, but expect [6].".format(
                                hero, action.shape
                            )
                        )
                        assert False
                    action_raw = np.array(action)
                    action = np.split(action_raw, format_shape, axis=1)
                else:
                    LOG.error(
                        "invalid action[{}] type of {}".format(hero, type(action))
                    )
                    assert False
                tmp_a = []
                for item in action:
                    tmp_a.append(tuple(item.reshape(item.shape[1])))

                each_camp_heros_action.append(tuple(tmp_a))
            rp_actions.append(tuple(each_camp_heros_action))
        return tuple(rp_actions)

    def _format_actions_one(self, actions):
        rp_actions = []
        for i, camp in enumerate(actions):
            each_camp_heros_action = []
            for hero, action in enumerate(camp):
                # formulation check
                if isinstance(action, (tuple, list)):
                    if not len(action) == 6:
                        LOG.error(
                            "action[{}] length incorrect: {}, but expect 6.".format(
                                hero, len(action)
                            )
                        )
                        assert False
                    action = np.array(action)
                elif isinstance(action, np.ndarray):
                    if not (len(action.shape) == 1 and action.shape[0] == 6):
                        LOG.error(
                            "action[{}] shape incorrect: {}, but expect [6].".format(
                                hero, action.shape
                            )
                        )
                        assert False
                else:
                    LOG.error(
                        "invalid action[{}] type of {}".format(hero, type(action))
                    )
                assert False
            rp_actions.append(tuple(each_camp_heros_action))
        return tuple(rp_actions)

    def _mask_players(self, rsp_pb):
        """
        Masked players will be controlled by common ai.
        """
        tmp_list = []
        for cmd in rsp_pb.cmd_list:
            if not self.player_masks.get(cmd.player_id):
                tmp_list.append(cmd)
        del rsp_pb.cmd_list[:]
        rsp_pb.cmd_list.extend(tmp_list)

        return rsp_pb

    @log_time("result_process")
    def _step_action(self, probs, agent_id):
        rp_actions = self._format_actions([probs])
        ret_code, resp_id = self.lib_processor.ResultProcess(
            rp_actions, self.cur_sgame_ids[agent_id]
        )

        if ret_code != int(interface.ReturnCode.PROCESS_ACTION_SUCCESS):
            raise Exception("process action failed: {}".format(ret_code))

        self._send(agent_id, ResponseType.CACHED, resp_id, self.cur_sgame_ids[agent_id])

        p_game_data = self.lib_processor.GetGameData(self.cur_sgame_ids[agent_id])
        if p_game_data is None:
            raise Exception("GetAIFrameState failed")

        return [x.data.game_state_info for x in p_game_data.result_process]

    @log_time("feature_process")
    def step_feature(self, agent_id):
        parse_state, sgame_id = self.lib_processor.RecvAIFrameState(
            self.addrs[agent_id]
        )
        if parse_state != int(
            interface.ReturnCode.PARSE_CONTINUE
        ) and parse_state != int(interface.ReturnCode.PARSE_SEND_EMPTY_ACTION):
            raise Exception("recv failed: %s", parse_state)

        ret_code, resp_id = self.lib_processor.FeatureProcess(parse_state, sgame_id)
        req_pb = None
        p_game_data = None
        if parse_state == int(interface.ReturnCode.PARSE_CONTINUE):
            p_game_data = self.lib_processor.GetGameData(sgame_id)
            if p_game_data is None:
                raise Exception("GetAIFrameState failed")

            req_pb = p_game_data.frame_state
            if req_pb is None:
                raise Exception("GetAIFrameState failed")

        self.cur_frame_no = req_pb.frame_no
        self.is_gameover = req_pb.gameover

        if ret_code == int(interface.ReturnCode.FEATURE_PROCESS_SEND_CACHED):
            # Note: Normal button up or none action for non-predict frame
            LOG.debug("send cached")
            self._send(agent_id, ResponseType.CACHED, resp_id, sgame_id)
            return 0, [], req_pb
        elif ret_code == int(interface.ReturnCode.FEATURE_PROCESS_SUCCESS):
            features = [state.data[0] for state in p_game_data.feature_process]
            return 1, features, req_pb
        elif ret_code == int(interface.ReturnCode.FEATURE_PROCESS_FAILED):
            raise Exception("FeatureProcess failed")
        elif ret_code == int(interface.ReturnCode.FEATURE_PROCESS_SEND_EMPTY):
            LOG.error("send empty")
            self._send_empty_rsp(agent_id)
            return 0, [], req_pb
        else:
            raise Exception("Unexpected return value: {}".format(ret_code))

    def _reset_gc_proc(self, agents, eval_mode=False):
        # connect to gc_proc
        LOG.info("Start a new game.")
        for i, agent in enumerate(agents):
            if agent.is_common_ai():
                port = 0
            else:
                port = 35300 + int(self.seed) * len(agents) + i
                if eval_mode:
                    port = port + 100
            addr = "tcp://0.0.0.0:{}".format(port)
            LOG.info("port %d addr %s" % (port, addr))
            self.addrs[i] = addr
            self.ports[i] = port

        # remote
        start_config = {
            "hero_conf": [],
        }
        for camp_id, hero_list in enumerate(next(self.camp_iter)):
            for idx, hero_id in enumerate(hero_list):
                hero_id = int(hero_id)
                if not self.ports[camp_id]:
                    request_info = {}
                elif idx == 0:
                    request_info = {
                        "ip": self.host,
                        "port": self.ports[camp_id],
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
            "runtime_id": self.runtime_id,
        }

        self.send_http_request(
            self.launch_server, "stopGame", self.user_token, data, ignore_resp=True
        )

        for i, agent in enumerate(agents):
            if agent.is_common_ai():
                continue

            addr = self.addrs[i]
            if not self.lib_processor.server_manager.Has(addr):
                for j in range(self.num_retry):
                    zmq_server = None
                    try:
                        zmq_server = self.lib_processor.server_manager.Add(addr)
                        if not zmq_server:
                            raise Exception("Address already exists: {}".format(addr))
                        rc = zmq_server.Reset(addr)
                        if rc < 0:
                            raise Exception(
                                "zmq_server.Reset failed: %s, retry %d/%d"
                                % (rc, j + 1, self.num_retry)
                            )
                        break
                    except Exception:
                        if zmq_server:
                            zmq_server.Close()
                        self.lib_processor.server_manager.Delete(addr)
                        time.sleep(1)
                        if j == self.num_retry - 1:
                            raise

        data = {
            "simulator_type": "remote_repeat",
            "runtime_id": self.runtime_id,
            "simulator_config": start_config,
        }
        self.send_http_request(self.launch_server, "newGame", self.user_token, data)

    def close_game(self, agents):
        """
        close game by sending signals to gamecore
        """
        while not self.is_gameover:
            LOG.error("game not end, send close game at first: %s" % self.cur_frame_no)
            for i, agent in enumerate(agents):
                if agent.is_common_ai():
                    continue

                parse_state, sgame_id = self.lib_processor.RecvAIFrameState(
                    self.addrs[i]
                )
                if parse_state != int(
                    interface.ReturnCode.PARSE_CONTINUE
                ) and parse_state != int(interface.ReturnCode.PARSE_SEND_EMPTY_ACTION):
                    raise Exception("recv failed: %s" % parse_state)

                self.lib_processor.FeatureProcess(parse_state, sgame_id)
                req_pb = None
                p_game_data = None
                if parse_state == int(interface.ReturnCode.PARSE_CONTINUE):
                    p_game_data = self.lib_processor.GetGameData(sgame_id)
                    if p_game_data is None:
                        raise Exception("GetAIFrameState failed")

                    req_pb = p_game_data.frame_state
                    if req_pb is None:
                        raise Exception("GetAIFrameState failed")

                LOG.debug(
                    "frame_no: {}, gameover: {}".format(
                        req_pb.frame_no, req_pb.gameover
                    )
                )

                self.is_gameover = req_pb.gameover
                self._gameover(i)

    def reset(self, agents, eval_mode=False):
        # reset infos
        LOG.debug("gcc reset: processor.reset")

        # reset infos
        sgame_ids = []
        if self.cur_sgame_ids:
            for sgame_id in self.cur_sgame_ids:
                if sgame_id:
                    sgame_ids.append(sgame_id)
        self.lib_processor.Reset(sgame_ids)
        self.cur_sgame_ids = [None] * self.num_player

        # eval
        self.lib_processor.SetEvalMode(eval_mode)

        self.last_predict_frame = 0
        self.player_masks = {}
        self.player_ids = {}
        self.player_camp = {}

        self._reset_gc_proc(agents, eval_mode)

        # get camp, game_id, gameover
        camp_game_id = []
        camp_gameover = []
        self.is_gameover = True
        is_gameover = 1
        for i, agent in enumerate(agents):
            LOG.info(
                "recv first msg, agent:%d is_common_ai:%d"
                % (i, (int)(agent.is_common_ai()))
            )
            if agent.is_common_ai():
                continue
            # length, req_type, seq_no, obs = self._recvmsg(agent_id=i)
            parse_state, sgame_id = self.lib_processor.RecvAIFrameState(self.addrs[i])
            if parse_state != int(
                interface.ReturnCode.PARSE_CONTINUE
            ) and parse_state != int(interface.ReturnCode.PARSE_SEND_EMPTY_ACTION):
                raise Exception("recv failed: %s" % parse_state)
            self.lib_processor.FeatureProcess(parse_state, sgame_id)
            req_pb = None
            p_game_data = None
            if parse_state == int(interface.ReturnCode.PARSE_CONTINUE):
                p_game_data = self.lib_processor.GetGameData(sgame_id)
                if p_game_data is None:
                    raise Exception("GetAIFrameState failed")

                req_pb = p_game_data.frame_state
                if req_pb is None:
                    raise Exception("GetAIFrameState failed")

            camp_game_id.append(bytes(req_pb.sgame_id, encoding="utf8"))
            camp_gameover.append(req_pb.gameover)
            self.cur_sgame_ids[i] = req_pb.sgame_id
            self._send_empty_rsp(agent_id=i)
        is_gameover = sum(camp_gameover)
        LOG.info(
            "camp_game_id:%s camp_gameover:%s is_gameover:%s"
            % (str(camp_game_id), str(camp_gameover), str(is_gameover))
        )
        return camp_game_id, is_gameover

    def _send_empty_rsp(self, agent_id):
        return self._send(agent_id, ResponseType.EMPTY, -1, None)

    def _send(self, agent_id, send_type, msg_id, sgame_id):
        ret = None
        if send_type == ResponseType.GAMEOVER:
            ret = self.lib_processor.SendGameoverResp(self.addrs[agent_id], sgame_id)
        elif send_type == ResponseType.EMPTY:
            ret = self.lib_processor.SendEmptyResp(self.addrs[agent_id])
        elif send_type == ResponseType.CACHED:
            ret = self.lib_processor.SendResp(self.addrs[agent_id], sgame_id, msg_id)
        else:
            raise Exception("Unknown ResponseType: %s" % send_type)

        if ret != int(interface.ReturnCode.SEND_SUCCESS):
            raise Exception("Send resp failed: %s" % ret)

        return True

    def send_http_request(
        self,
        server_addr,
        req_type,
        token,
        data,
        download_path=None,
        no_python=False,
        ignore_resp=False,
    ):
        if server_addr is None:
            server_addr = "127.0.0.1:23432"
        url = "http://%s/v2/%s" % (server_addr, req_type)
        headers = {
            "Content-Type": "application/json",
        }

        LOG.info(str(data))

        if download_path is not None:
            data["Path"] = download_path
            r = requests.post(
                url=url,
                json=data,
                headers=headers,
                verify=False,
                stream=True,
            )
            try:
                r.raise_for_status()
                return r
            except Exception as e:
                LOG.ERROR("[Warning] download file {} failed.".format(download_path))
                traceback.print_exc()

        else:
            if no_python:
                curl_command = "curl -k {} -d '{}'".format(url, json.dumps(data))
                LOG.info("curl_command", curl_command)
                os.system(curl_command)
                return
            else:
                resp = requests.post(url=url, json=data, headers=headers, verify=False)
                if resp.ok:
                    return resp.content if ignore_resp else resp.json()
                else:
                    return {}
