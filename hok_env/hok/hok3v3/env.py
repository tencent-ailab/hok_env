# -*- coding: utf-8 -*-

import numpy as np

from hok.common.log import log_time
from hok.common.log import logger as LOG


class Environment:
    def __init__(
        self,
        aiservers,
        lib_processor,
        game_launcher,
        runtime_id,
        wait_game_max_timeout=30,
        aiserver_ip="127.0.0.1",
    ):
        self.aiservers = aiservers
        self.lib_processor = lib_processor
        self.game_launcher = game_launcher
        self.runtime_id = runtime_id
        self.wait_game_max_timeout = wait_game_max_timeout
        self.aiserver_ip = aiserver_ip

        self.cur_sgame_ids = []

    @log_time("feature_process")
    def step_feature(self, agent_id):
        return self.aiservers[agent_id].recv_and_feature_process()

    @log_time("result_process")
    def step_action(self, agent_id, probs, features, frame_state):
        return self.aiservers[agent_id].result_process(probs, features, frame_state)

    def close_game(self, force=False):
        if not force:
            # wait game over
            self.game_launcher.wait_game(self.runtime_id, self.wait_game_max_timeout)

        # force close
        self.game_launcher.stop_game(self.runtime_id)

    def _get_server(self, use_common_ai):
        return [
            None
            if use_common_ai[i]
            else (self.aiserver_ip, int(server.addr.split(":")[-1]))
            for i, server in enumerate(self.aiservers)
        ]

    # TODO gym style
    def reset(
        self, use_common_ai, camp_hero_list, eval_mode=False, extra_abs_key_info=None
    ):
        """
        extra_abs_key_info: 见start_game的extra_abs_key_info字段
        """
        LOG.debug("reset env")
        # reset infos
        self.lib_processor.Reset(self.cur_sgame_ids)
        self.cur_sgame_ids.clear()

        # eval_mode
        self.lib_processor.SetEvalMode(eval_mode)

        # stop game & start server & start game
        self.game_launcher.stop_game(self.runtime_id)

        for i, is_common_ai in enumerate(use_common_ai):
            if is_common_ai:
                continue
            self.aiservers[i].start()

        self.game_launcher.start_game(
            self.runtime_id,
            self._get_server(use_common_ai),
            camp_hero_list,
            eval_mode=eval_mode,
            extra_abs_key_info=extra_abs_key_info,
        )

        # process first frame
        for i, is_common_ai in enumerate(use_common_ai):
            LOG.info(f"Reset info: agent:{i} is_common_ai:{is_common_ai}")
            if is_common_ai:
                continue
            continue_process, features, frame_state = self.aiservers[
                i
            ].recv_and_feature_process()
            self.cur_sgame_ids.append(frame_state.sgame_id)

            if continue_process:
                self.aiservers[i].send_empty_rsp()
