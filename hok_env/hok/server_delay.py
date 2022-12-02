import logging

import hok.lib.interface as interface

from hok.server import AIServer as AIServerBase
from hok.server import ResponceType, default_config_path

LOG = logging.getLogger(__file__)


class AIServer(AIServerBase):
    def __init__(self, agent, addr, config_path=default_config_path) -> None:
        super().__init__(agent, addr, config_path)
        self.last = {}

    def clear_game(self, sgame_id):
        super().clear_game(sgame_id)
        self.last.pop(sgame_id, None)

    def process(self):
        parse_state, sgame_id = self.lib_processor.RecvAIFrameState(self.addr)
        self._send_last(sgame_id)

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
                self._put_empty_rsp(sgame_id)
            elif ret[0] == 4:
                self._put(ResponceType.NONE, -1, sgame_id)
            elif ret[0] == 5:
                # 初始化随机动作
                self._put(ResponceType.CACHED, int(ret[1]), sgame_id)
            return
        else:
            LOG.error("Unexpected return value: {}".format(ret[0]))
            return

        if req_pb.gameover:
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

            ret_code, resp_id = self.lib_processor.ResultProcess(rp_actions, sgame_id)
            if ret_code != interface.PROCESS_ACTION_SUCCESS:
                LOG.warn("process action failed: {}".format(ret_code))
                return

            self._put(ResponceType.CACHED, resp_id, sgame_id)

    def _put_empty_rsp(self, sgame_id):
        return self._put(ResponceType.DEFAULT, -1, sgame_id)

    def _put(self, send_type, msg_id, sgame_id):
        self.last[sgame_id] = (send_type, msg_id, sgame_id)

    def _send_last(self, sgame_id):
        last = self.last.pop(sgame_id, None)
        if not last:
            # 无缓存的上一次预测结果
            self._send_empty_rsp()
            return

        send_type, msg_id, sgame_id = last
        ret = self._send(send_type, msg_id, sgame_id)
        if not ret:
            LOG.warn("send failed, send empty rsp: {}".format(ret))
            self._send_empty_rsp()
