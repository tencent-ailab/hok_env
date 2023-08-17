import os
import time
from enum import Enum

import numpy as np

from hok.common.log import log_time
import hok.common.log as LOG

from hok.hok3v3.action_space import DumpProbs
import hok.hok3v3.lib.lib3v3 as interface  # TODO 是否有办法剥离? 或者通过lib_processor访问?


class ResponseType(Enum):
    EMPTY = 1
    CACHED = 2
    GAMEOVER = 3


class AIServer:
    def __init__(
        self,
        addr,
        lib_processor,
    ) -> None:
        self.addr = addr
        self.lib_processor = lib_processor
        self.num_retry = 5

    def start(self):
        LOG.info(f"Start server at {self.addr}")
        if self.lib_processor.server_manager.Has(self.addr):
            LOG.info(f"Server already exists at {self.addr}, skip")
            return
        for j in range(self.num_retry):
            zmq_server = None
            try:
                zmq_server = self.lib_processor.server_manager.Add(self.addr)
                if not zmq_server:
                    raise Exception(f"Failed to add server at: {self.addr}")
                rc = zmq_server.Reset(self.addr)
                if rc < 0:
                    raise Exception(
                        f"zmq_server.Reset failed: {rc}, retry {j+1}/{self.num_retry}"
                    )
                return
            except Exception:
                if zmq_server:
                    zmq_server.Close()
                self.lib_processor.server_manager.Delete(self.addr)
                time.sleep(1)
                if j == self.num_retry - 1:
                    raise

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
            LOG.warning("Unknown ResponseType: %s" % send_type)
            return False

        if ret != int(interface.ReturnCode.SEND_SUCCESS):
            LOG.warning("Send resp failed: %s" % ret)
            return False

        return True

    def send_empty_rsp(self):
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
                            f"action[{hero}] length incorrect: {len(action)}, but expect {data_len}."
                        )

                        return False, None
                    action_raw = np.array(action)
                    action = np.split(action_raw, format_shape, axis=1)
                elif isinstance(action, np.ndarray):
                    # if not (len(action.shape) == 1 and action.shape[0] == 6):
                    if not (action.shape[1] == data_len):
                        LOG.error(
                            f"action[{hero}] shape incorrect: {action.shape}, but expect [{data_len}]."
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

    def recv_and_feature_process(self):
        """
        feature_process: process game state

        :return: continue_process flag and p_game_data
        """
        parse_state, sgame_id = self.lib_processor.RecvAIFrameState(self.addr)
        sent = False
        continue_process = False
        p_game_data = None
        try:
            if parse_state != int(interface.ReturnCode.PARSE_CONTINUE):
                raise Exception("recv failed: %s" % parse_state)

            ret_code, resp_id = self.lib_processor.FeatureProcess(parse_state, sgame_id)

            # 非预测帧也需要能够获取状态, 比如gameover
            p_game_data = self.lib_processor.GetGameData(sgame_id)
            if p_game_data is None:
                raise Exception("GetAIFrameState failed: p_game_data is None")

            req_pb = p_game_data.frame_state
            if req_pb is None:
                raise Exception("GetAIFrameState failed: req_pb is None")

            if ret_code == int(interface.ReturnCode.FEATURE_PROCESS_SEND_CACHED):
                # Note: Normal button up or none action for non-predict frame
                LOG.debug("FEATURE_PROCESS_SUCCESS send cached")
                sent = self._send(ResponseType.CACHED, resp_id, sgame_id)
            elif ret_code == int(interface.ReturnCode.FEATURE_PROCESS_SUCCESS):
                continue_process = True
            else:
                raise Exception("Unexpected return value: {}".format(ret_code))
        finally:
            # 在冲突退出, 或者其他错误情况, 回空包以保证zmq的状态机正确
            if not sent and not continue_process:
                LOG.error("not sent, send empty rsp")
                self.send_empty_rsp()

        return continue_process, p_game_data

    def result_process(self, probs, p_game_data):
        ok, rp_actions = self._format_actions([probs])
        if not ok:
            raise Exception("format action failed")

        req_pb = p_game_data.frame_state

        ret_code, resp_id = self.lib_processor.ResultProcess(
            rp_actions, req_pb.sgame_id
        )
        if ret_code != int(interface.ReturnCode.PROCESS_ACTION_SUCCESS):
            raise Exception("process action failed: {}".format(ret_code))

        LOG.debug("ResultProcess send cached")
        return self._send(ResponseType.CACHED, resp_id, req_pb.sgame_id)


class BattleServer(AIServer):
    def __init__(
        self,
        agent,
        addr,
        lib_processor,
        enable_dump_probs=os.getenv("DUMP_PROBS") == "1",
        dump_probs_dir="/aiarena/logs/probs/",
    ) -> None:
        super().__init__(addr, lib_processor)
        self.agent = agent
        self.enable_dump_probs = enable_dump_probs
        self.dump_probs_dir = dump_probs_dir

    def clear_game(self, sgame_id):
        self.lib_processor.Reset([sgame_id])

    def _feature(self, p_game_data):
        return [state.data[0] for state in p_game_data.feature_process]

    def _result(self, p_game_data):
        return [x.data.game_state_info for x in p_game_data.result_process]

    def run(self):
        LOG.info("socket addr %s" % (self.addr))
        zmq_server = self.lib_processor.server_manager.Add(self.addr)
        rc = zmq_server.Reset(self.addr)
        if rc < 0:
            raise Exception("zmq_server.Reset failed")

        while True:
            try:
                self.process()
            except Exception:
                LOG.exception("process failed")
                while True:
                    time.sleep(1)
                    rc = zmq_server.Reset(self.addr)
                    if rc < 0:
                        LOG.exception("zmq_server.Reset failed")
                        continue
                    break

    def process(self):
        continue_process, p_game_data = self.recv_and_feature_process()

        req_pb = p_game_data.frame_state

        if continue_process:
            features = self._feature(p_game_data)
            sent = False
            try:
                # TODO hongyangqin save lstm_info
                probs, lstm_info = self.agent.predict_process(features, req_pb)
                sent = self.result_process(probs, p_game_data)

                if self.enable_dump_probs:
                    try:
                        DumpProbs(
                            req_pb, features, self._result(p_game_data)
                        ).save_to_file(
                            os.path.join(
                                self.dump_probs_dir, "{}.bin".format(req_pb.sgame_id)
                            )
                        )
                    except Exception:
                        LOG.exception("dump probs failed")
            finally:
                if not sent:
                    LOG.warning("not sent, send empty rsp2")
                    self.send_empty_rsp()

        if req_pb.gameover:
            LOG.info("game done: {}, {}".format(req_pb.sgame_id, req_pb.frame_no))
            # 释放旧sgame_id的资源
            self.clear_game(req_pb.sgame_id)
