import os
import time
import logging
from enum import Enum

import numpy as np

import hok.lib.interface as interface

LOG = logging.getLogger(__file__)

default_config_path = os.path.join(os.path.dirname(__file__), "config.dat")


class ResponceType(Enum):
    NONE = 0
    DEFAULT = 1
    CACHED = 2
    GAMEOVER = 3


class AIServer:
    def __init__(self, agent, addr, config_path=default_config_path) -> None:
        self.agent = agent
        self.addr = addr
        self.lib_processor = interface.Interface()
        self.lib_processor.Init(config_path)
        self.zmq_server = None
        self.action_size = [12, 16, 16, 16, 16, 8]
        self.HERO_ID_INDEX_DICT = {
            112: 0,
            121: 1,
            123: 2,
            131: 3,
            132: 4,
            133: 5,
            140: 6,
            141: 7,
            146: 8,
            150: 9,
            154: 10,
            157: 11,
            163: 12,
            169: 13,
            175: 14,
            182: 15,
            193: 16,
            199: 17,
            502: 18,
            513: 19,
        }
        self.lstm_info = {}

    def _save_lstm_info(self, agent, sgame_id):
        lstm_info = agent.get_lstm_info()
        self.lstm_info[sgame_id] = lstm_info

    def _restore_lstm_info(self, agent, sgame_id):
        lstm_info = self.lstm_info.get(sgame_id)

        # no lstm info
        if lstm_info is None:
            return

        agent.set_lstm_info(lstm_info)

    def clear_game(self, sgame_id):
        self.lib_processor.Reset(True, [sgame_id])
        self.lstm_info.pop(sgame_id, None)

    def run(self):
        LOG.info("socket addr %s" % (self.addr))
        self.zmq_server = self.lib_processor.server_manager.Add(self.addr)
        rc = self.zmq_server.Reset(self.addr)
        if rc < 0:
            raise Exception("zmq_server.Reset failed")

        while True:
            try:
                self.process()
            except Exception:
                LOG.exception("process failed")
                while True:
                    time.sleep(1)
                    rc = self.zmq_server.Reset(self.addr)
                    if rc < 0:
                        LOG.exception("zmq_server.Reset failed")
                        continue
                    break

    def _state_tuple2np(self, states, hero_id):
        states = list(states)
        for state in states:
            if state is None:
                continue
            for k in state:
                if isinstance(state[k], tuple) and k in ["legal_action"]:
                    state[k] = np.array(state[k])
                if isinstance(state[k], tuple) and k in ["observation"]:
                    state[k] = np.array(state[k])
                    hero_id_vec = np.zeros(
                        [
                            len(self.HERO_ID_INDEX_DICT),
                        ],
                        dtype=np.float,
                    )
                    hero_id_vec[self.HERO_ID_INDEX_DICT[hero_id]] = 1

                    state[k] = np.concatenate((state[k], hero_id_vec), axis=0)
                if isinstance(state[k], dict) and k in ["sub_action_mask"]:
                    for i in state[k]:
                        state[k][i] = np.array(state[k][i])

        return states

    def _format_actions(self, actions):
        # check whether the actions are within defined range, and format into gamecore actions
        rp_actions = []
        for i, action in enumerate(actions):
            # formulation check
            if isinstance(action, (tuple, list)):
                if not len(action) == 6:
                    LOG.warn("action[{}] length incorrect: {}, but expect 6.")
                    return None
                action = np.array(action)
            elif isinstance(action, np.ndarray):
                if not (len(action.shape) == 1 and action.shape[0] == 6):
                    LOG.warn(
                        "action[{}] shape incorrect: {}, but expect [6].".format(
                            i, action.shape
                        )
                    )
                    return None
            else:
                LOG.warn("invalid action[{}] type of {}".format(i, type(action)))
                return None

            old_action = action
            action = []
            for j, act in enumerate(old_action):
                if not (0 <= act < self.action_size[j]):
                    LOG.warn(
                        "Action[{}] {}: {} not in [0,{})".format(
                            i, j, act, self.action_size[j]
                        )
                    )
                    return None
                action.append((act,))
            action = tuple(action)
            rp_act = (
                (
                    (0,) * 12,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 8,
                )
                + action
                + ((0,),)
            )
            rp_actions.append(rp_act)
        return tuple(rp_actions)

    def process(self):
        parse_state, sgame_id = self.lib_processor.RecvAIFrameState(self.addr)
        sent = False
        try:
            if (
                parse_state != interface.PARSE_CONTINUE
                and parse_state != interface.PARSE_NONE_ACTION
            ):
                LOG.warn("recv failed: %s", parse_state)
                return

            req_pb = None
            if parse_state == interface.PARSE_CONTINUE:
                req_pb = self.lib_processor.GetAIFrameState(sgame_id)
                if req_pb is None:
                    LOG.warn("GetAIFrameState failed")
                    return

            ret = self.lib_processor.FeatureProcess(parse_state, sgame_id)
            # Failed, return no action
            if ret[0] == 0:
                LOG.error("step failed: {}".format(ret[1]))
                return
            if ret[0] == 1:
                LOG.error("Parsing gameover information, receive msg again!")
                return
            elif ret[0] == 2:
                # continue to result_process
                # SEND_CCD_ONE_HERO, get normal feature vector, break
                state = self._state_tuple2np(ret[1:], req_pb.hero_list[0].config_id)[0]
                state["req_pb"] = req_pb
                state["sgame_id"] = sgame_id
            elif ret[0] == 3 or ret[0] == 4 or ret[0] == 5:
                # SEND_CCD_FIVE_HERO
                if ret[0] == 3:
                    sent = self._send_empty_rsp()
                elif ret[0] == 4:
                    sent = self._send(ResponceType.NONE, -1, sgame_id)
                elif ret[0] == 5:
                    # 初始化随机动作
                    sent = self._send(ResponceType.CACHED, int(ret[1]), sgame_id)
                return
            else:
                LOG.error("Unexpected return value: {}".format(ret[0]))
                return

            if req_pb.gameover:
                sent = self._send_empty_rsp()
                LOG.info("game done: {}, {}".format(sgame_id, req_pb.frame_no))
                # 释放旧sgame_id的资源
                self.clear_game(sgame_id)
            else:
                self._restore_lstm_info(self.agent, sgame_id)
                _, d_action, _ = self.agent.process(state)
                self._save_lstm_info(self.agent, sgame_id)

                rp_actions = self._format_actions([d_action])
                if not rp_actions:
                    return

                ret_code, resp_id = self.lib_processor.ResultProcess(
                    rp_actions, sgame_id
                )
                if ret_code != interface.PROCESS_ACTION_SUCCESS:
                    LOG.warn("process action failed: {}".format(ret_code))
                    return

                sent = self._send(ResponceType.CACHED, resp_id, sgame_id)
        finally:
            # 在冲突退出, 或者其他错误情况, 回空包以保证zmq的状态机正确
            if not sent:
                LOG.warn("not sent, send empty rsp")
                self._send_empty_rsp()

    def _send_empty_rsp(self):
        return self._send(ResponceType.DEFAULT, -1, None)

    def _send(self, send_type, msg_id, sgame_id):
        ret = None
        if send_type == ResponceType.GAMEOVER:
            ret = self.lib_processor.SendGameoverResp(self.addr, sgame_id)
        elif send_type == ResponceType.DEFAULT:
            ret = self.lib_processor.SendDefaultResp(self.addr)
        elif send_type == ResponceType.NONE:
            ret = self.lib_processor.SendNoneResp(self.addr, sgame_id)
        elif send_type == ResponceType.CACHED:
            ret = self.lib_processor.SendResp(self.addr, sgame_id, msg_id)
        else:
            LOG.warn("Unknown ResponceType: %s" % send_type)
            return False

        if ret != interface.SEND_SUCCESS:
            LOG.warn("Send resp failed: %s" % ret)
            return False

        return True
