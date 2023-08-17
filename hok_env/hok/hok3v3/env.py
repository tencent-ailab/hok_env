# -*- coding: utf-8 -*-

import numpy as np

from hok.common.log import log_time
import hok.common.log as LOG


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

    def _feature(self, p_game_data):
        return [state.data[0] for state in p_game_data.feature_process]

    def _result(self, p_game_data):
        return [x.data.game_state_info for x in p_game_data.result_process]

    @log_time("feature_process")
    def step_feature(self, agent_id):
        continue_process, p_game_data = self.aiservers[
            agent_id
        ].recv_and_feature_process()

        if continue_process:
            return continue_process, self._feature(p_game_data), p_game_data
        else:
            return continue_process, [], p_game_data

    @log_time("result_process")
    def step_action(self, agent_id, features, probs, p_game_data, lstm_info):
        self.aiservers[agent_id].result_process(probs, p_game_data)
        return self._sample_process(features, p_game_data, lstm_info)

    def _sample_process(self, features, p_game_data, lstm_info):
        req_pb = p_game_data.frame_state
        process_result = self._result(p_game_data)

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

    # TODO hongyangqin gym style
    def reset(self, use_common_ai, camp_hero_list, eval_mode=False):
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
        )

        # process first frame
        for i, is_common_ai in enumerate(use_common_ai):
            LOG.info(f"Reset info: agent:{i} is_common_ai:{is_common_ai}")
            if is_common_ai:
                continue
            continue_process, p_game_data = self.aiservers[i].recv_and_feature_process()
            req_pb = p_game_data.frame_state
            self.cur_sgame_ids.append(req_pb.sgame_id)

            if continue_process:
                self.aiservers[i].send_empty_rsp()
