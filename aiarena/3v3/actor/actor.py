# -*- coding: utf-8 -*-
"""
    KingHonour Data production process
"""
import os
import time

from rl_framework.common.logging import g_log_time, log_time, log_time_func
import rl_framework.common.logging as LOG

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Actor:
    def __init__(
        self,
        id,
        agents,
        env,
        sample_manager,
        camp_iter,
        max_episode=-1,
        monitor_logger=None,
        send_sample_frame=963,
    ):
        self.m_task_uuid = "TODO TASK_UUID"
        self.m_config_id = id

        self.agents = agents
        self.env = env
        self.sample_manager = sample_manager
        self.camp_iter = camp_iter

        self._max_episode = max_episode
        self.monitor_logger = monitor_logger
        self.send_sample_frame = send_sample_frame

        self._episode_num = 0

    @log_time("one_episode")
    def _run_episode(self, camp_config):
        LOG.info("Start a new game")
        g_log_time.clear()

        log_time_func("reset")
        # swap two agents, and re-assign camp
        self.agents.reverse()

        for i, agent in enumerate(self.agents):
            LOG.debug("reset agent {}".format(i))
            agent.reset()

        # restart a new game
        use_common_ai = [agent.is_common_ai() for agent in self.agents]
        self.env.reset(use_common_ai, camp_config)
        first_frame_no = -1

        # reset mem pool and models
        self.sample_manager.reset()
        log_time_func("reset", end=True)

        game_info = {}
        is_gameover = False
        req_pb = None
        while not is_gameover:
            log_time_func("one_frame")

            continue_process = False
            # while True:
            is_send = False
            reward_camp = [[], []]
            all_hero_reward_camp = [[], []]
            for i, agent in enumerate(self.agents):
                if use_common_ai[i]:
                    LOG.debug(f"agent {i} is common_ai")
                    continue

                continue_process, features, p_game_data = self.env.step_feature(i)
                req_pb = p_game_data.frame_state
                if req_pb.gameover:
                    game_info["length"] = req_pb.frame_no
                    is_gameover = True

                if not continue_process:
                    continue

                if first_frame_no < 0:
                    first_frame_no = req_pb.frame_no
                    LOG.info("first_frame_no %d" % first_frame_no)

                for hero_idx in range(len(features)):
                    reward_camp[i].append(features[hero_idx].reward)
                    all_hero_reward_camp[i].append(
                        features[hero_idx].all_hero_reward_info
                    )

                sample = {}
                probs, lstm_info = agent.predict_process(features, req_pb)
                sample = self.env.step_action(
                    i, features, probs, p_game_data, lstm_info
                )

                # skip save sample if not latest model
                if not agent.is_latest_model:
                    continue

                is_send = req_pb.gameover or (
                    ((req_pb.frame_no - first_frame_no) % self.send_sample_frame == 0)
                    and (req_pb.frame_no > first_frame_no)
                )

                if not is_send:
                    self.sample_manager.save_sample(
                        **sample, agent_id=i, uuid=self.m_task_uuid
                    )
                else:
                    LOG.info(
                        f"save_last_sample frame[{req_pb.frame_no}] req_pb.gameover[{req_pb.gameover}]"
                    )
                    if req_pb.gameover:
                        self.sample_manager.save_last_sample(
                            agent_id=i,
                            reward=reward_camp[i],
                            all_hero_reward_s=all_hero_reward_camp[i],
                            # TODO hongyangqin refactor reward_manager
                        )
                    else:
                        self.sample_manager.save_last_sample(
                            agent_id=i,
                            reward=reward_camp[i],
                            value_s=sample["value_s"],
                            all_hero_reward_s=all_hero_reward_camp[i],
                        )

            if is_send or is_gameover:
                LOG.info("send_sample and update model")
                self.sample_manager.send_samples()
                self.sample_manager.reset()
                for i, agent in enumerate(self.agents):
                    agent.update_model()
                LOG.info("send_sample and update model done.")
        log_time_func("one_frame", end=True)
        self.env.close_game()

        if not req_pb:
            return

        # process game info
        loss_camp = None
        # update camp information.
        for organ in req_pb.organ_list:
            if organ.type == 24 and organ.hp <= 0:
                loss_camp = organ.camp

            if organ.type in [21, 24]:
                LOG.info(
                    "Tower {} in camp {}, hp: {}".format(
                        organ.type, organ.camp, organ.hp
                    )
                )

        for i, agent in enumerate(self.agents):
            agent_camp = i + 1
            agent_win = 0
            if (loss_camp is not None) and (agent_camp != loss_camp):
                agent_win = 1
            LOG.info("camp%d_agent:%d win:%d" % (agent_camp, i, agent_win))

            LOG.info("---------- camp%d hero_info ----------" % agent_camp)
            for hero_state in req_pb.hero_list:
                if agent_camp != hero_state.camp:
                    continue

                LOG.info(
                    "hero:%d moneyCnt:%d killCnt:%d deadCnt:%d assistCnt:%d"
                    % (
                        hero_state.config_id,
                        hero_state.moneyCnt,
                        hero_state.killCnt,
                        hero_state.deadCnt,
                        hero_state.assistCnt,
                    )
                )

        if self.m_config_id == 0:
            for i, agent in enumerate(self.agents):
                if not agent.keep_latest:
                    continue

                money_per_frame = 0
                kill = 0
                death = 0
                assistCnt = 0
                hurt_per_frame = 0
                hurtH_per_frame = 0
                hurtBH_per_frame = 0

                agent_camp = i + 1
                agent_win = 0
                if (loss_camp is not None) and (agent_camp != loss_camp):
                    agent_win = 1

                hero_idx = 0
                for hero_state in req_pb.hero_list:
                    if agent_camp == hero_state.camp:
                        hero_idx += 1
                        money_per_frame += hero_state.moneyCnt / game_info["length"]
                        kill += hero_state.killCnt
                        death += hero_state.deadCnt
                        assistCnt += hero_state.assistCnt
                        hurt_per_frame += hero_state.totalHurt / game_info["length"]
                        hurtH_per_frame += (
                            hero_state.totalHurtToHero / game_info["length"]
                        )
                        hurtBH_per_frame += (
                            hero_state.totalBeHurtByHero / game_info["length"]
                        )

                game_info["money_per_frame"] = money_per_frame / hero_idx
                game_info["kill"] = kill / hero_idx
                game_info["death"] = death / hero_idx
                game_info["assistCnt"] = assistCnt / hero_idx
                game_info["hurt_per_frame"] = hurt_per_frame / hero_idx
                game_info["hurtH_per_frame"] = hurtH_per_frame / hero_idx
                game_info["hurtBH_per_frame"] = hurtBH_per_frame / hero_idx
                game_info["win"] = agent_win

            if self.monitor_logger:
                self.monitor_logger.info(game_info)

    def run(self):
        self._last_print_time = time.time()
        self._episode_num = 0
        MAX_REPEAT_ERR_NUM = 2
        repeat_num = MAX_REPEAT_ERR_NUM

        while True:
            try:
                camp_config = next(self.camp_iter)
                self._run_episode(camp_config)
                self._episode_num += 1
                repeat_num = MAX_REPEAT_ERR_NUM
            except Exception as e:
                LOG.exception(
                    "_run_episode err: {}/{}".format(repeat_num, MAX_REPEAT_ERR_NUM)
                )
                repeat_num -= 1
                if repeat_num == 0:
                    raise e

            if 0 < self._max_episode <= self._episode_num:
                break
