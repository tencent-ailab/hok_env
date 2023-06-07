import numpy as np
import os
import time
from enum import Enum

from hok.hok3v3.action_space import DumpProbs
from hok.hok3v3 import CONFIG_DAT
import hok.hok3v3.lib.lib3v3 as interface

from rl_framework.common.logging import log_time
import rl_framework.common.logging as LOG


class ResponseType(Enum):
    NONE = 0
    EMPTY = 1
    CACHED = 2
    GAMEOVER = 3


class AIServer:
    def __init__(
        self,
        agent,
        addr,
        enable_dump_probs=os.getenv("DUMP_PROBS") == "1",
        dump_probs_dir="/aiarena/logs/probs/",
    ) -> None:
        self.agent = agent
        self.addr = addr
        self.lib_processor = interface.Interface()
        self.lib_processor.Init(CONFIG_DAT)
        self.lib_processor.SetEvalMode(True)

        self.zmq_server = None
        self.enable_dump_probs = enable_dump_probs
        self.dump_probs_dir = dump_probs_dir

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

    def _feature(self, p_game_data):
        return [state.data[0] for state in p_game_data.feature_process]

    def _result(self, p_game_data):
        return [x.data.game_state_info for x in p_game_data.result_process]

    @log_time("send_data")
    def _send(self, send_type, msg_id, sgame_id):
        ret = None
        if send_type == ResponseType.GAMEOVER:
            ret = self.lib_processor.SendGameoverResp(self.addr, sgame_id)
        elif send_type == ResponseType.EMPTY:
            ret = self.lib_processor.SendEmptyResp(self.addr)
        elif send_type == ResponseType.CACHED:
            ret = self.lib_processor.SendResp(self.addr, sgame_id, msg_id)
        else:
            LOG.warn("Unknown ResponseType: %s" % send_type)
            return False

        if ret != int(interface.ReturnCode.SEND_SUCCESS):
            LOG.warn("Send resp failed: %s" % ret)
            return False

        return True

    def _send_empty_rsp(self):
        return self._send(ResponseType.EMPTY, -1, None)

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
                        return False, None
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
                        return False, None
                    action_raw = np.array(action)
                    action = np.split(action_raw, format_shape, axis=1)
                else:
                    LOG.error(
                        "invalid action[{}] type of {}".format(hero, type(action))
                    )
                    return False, None
                tmp_a = []
                for item in action:
                    tmp_a.append(tuple(item.reshape(item.shape[1])))

                each_camp_heros_action.append(tuple(tmp_a))
            rp_actions.append(tuple(each_camp_heros_action))
        return True, tuple(rp_actions)

    def process(self):
        parse_state, sgame_id = self.lib_processor.RecvAIFrameState(self.addr)
        sent = False
        try:
            if parse_state != int(
                interface.ReturnCode.PARSE_CONTINUE
            ) and parse_state != int(interface.ReturnCode.PARSE_SEND_EMPTY_ACTION):
                LOG.warn("recv failed: %s", parse_state)
                return

            req_pb = None
            p_game_data = None
            if parse_state == int(interface.ReturnCode.PARSE_CONTINUE):
                p_game_data = self.lib_processor.GetGameData(sgame_id)
                if p_game_data is None:
                    LOG.warn("GetAIFrameState failed")
                    return

                req_pb = p_game_data.frame_state
                if req_pb is None:
                    LOG.warn("GetAIFrameState failed")
                    return

            ret_code, resp_id = self.lib_processor.FeatureProcess(parse_state, sgame_id)
            if ret_code == int(interface.ReturnCode.FEATURE_PROCESS_SEND_CACHED):
                # Note: Normal button up or none action for non-predict frame
                LOG.debug("send cached")
                sent = self._send(ResponseType.CACHED, resp_id, sgame_id)
            elif ret_code == int(interface.ReturnCode.FEATURE_PROCESS_SUCCESS):
                sent = self._predict_frame(req_pb, sgame_id, p_game_data)
            elif ret_code == int(interface.ReturnCode.FEATURE_PROCESS_FAILED):
                LOG.error("step failed")
            elif ret_code == int(interface.ReturnCode.FEATURE_PROCESS_SEND_EMPTY):
                LOG.error("send empty")
                sent = self._send_empty_rsp()
            else:
                LOG.error("Unexpected return value: {}".format(ret_code))
        finally:
            # 在冲突退出, 或者其他错误情况, 回空包以保证zmq的状态机正确
            if not sent:
                LOG.warn("not sent, send empty rsp")
                self._send_empty_rsp()

    def _predict_frame(self, req_pb, sgame_id, p_game_data):
        features = self._feature(p_game_data)
        sent = False
        if req_pb.gameover:
            sent = self._send_empty_rsp()
            LOG.info("game done: {}, {}".format(req_pb.sgame_id, req_pb.frame_no))
            # 释放旧sgame_id的资源
            self.clear_game(sgame_id)
        else:
            probs, _ = self.agent.predict_process(features, req_pb)

            ok, rp_actions = self._format_actions([probs])
            if not ok:
                return sent

            ret_code, resp_id = self.lib_processor.ResultProcess(rp_actions, sgame_id)
            if ret_code != int(interface.ReturnCode.PROCESS_ACTION_SUCCESS):
                LOG.error("process action failed: {}".format(ret_code))
                return sent

            sent = self._send(ResponseType.CACHED, resp_id, sgame_id)

            if self.enable_dump_probs:
                try:
                    DumpProbs(req_pb, features, self._result(p_game_data)).save_to_file(
                        os.path.join(
                            self.dump_probs_dir, "{}.bin".format(req_pb.sgame_id)
                        )
                    )
                except Exception:
                    LOG.exception("dump probs failed")

        return sent

    def clear_game(self, sgame_id):
        self.lib_processor.Reset([sgame_id])
