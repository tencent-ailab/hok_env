# -*- coding: utf-8 -*-
"""
    KingHonour Data production process
"""
import os
import traceback
from collections import deque
import time

import numpy as np
from config.config import Config

from rl_framework.common.logging import g_log_time, log_time_func
import rl_framework.common.logging as LOG

IS_TRAIN = Config.IS_TRAIN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
OS_ENV = os.environ
IS_DEV = OS_ENV.get("IS_DEV")


class Actor:
    """
    used for sample logic
        run 1 episode
        save sample in sample manager
    """

    def __init__(self, id, agents, max_episode: int = 0, env=None, monitor_logger=None):
        self.m_config_id = id
        self.m_task_uuid = Config.TASK_UUID
        self.m_steps = Config.STEPS
        self.m_init_path = Config.INIT_PATH
        self.m_update_path = Config.UPDATE_PATH
        self.m_mem_pool_path = Config.MEM_POOL_PATH
        self.m_task_name = Config.TASK_NAME

        # self.m_replay_buffer = deque()
        self.m_episode_info = deque(maxlen=100)
        # self.m_ip = CommonFunc.get_local_ip()
        self.env = env
        self._max_episode = max_episode

        self.m_run_step = 0
        self.m_best_reward = 0

        self._last_print_time = time.time()
        self._episode_num = 0

        self.agents = agents
        self.hero_num = 3
        self.send_sample_frame = Config.SEND_SAMPLE_FRAME
        self.monitor_logger = monitor_logger

    def set_env(self, environment):
        self.env = environment

    def set_sample_managers(self, sample_manager):
        self.m_sample_manager = sample_manager

    def is_crystal_gameover(self, req_pb):
        crystal_gameover = 0
        for npc_state in req_pb.frame_state.npc_states:
            if (
                npc_state.actor_type == 2
                and npc_state.sub_type == 24
                and npc_state.hp <= 0
            ):
                crystal_gameover = 1
                LOG.info(
                    "Tower %d in camp %d, hp: %d"
                    % (npc_state.sub_type, npc_state.camp, npc_state.hp)
                )
        return crystal_gameover

    def set_agents(self, agents):
        self.agents = agents

    def _run_episode(self, eval=False, load_models=None, eval_info=""):
        time.sleep(5)
        for item in g_log_time.items():
            g_log_time[item[0]] = []
        # alias
        sample_manager = self.m_sample_manager

        done = False

        log_time_func("reset")
        log_time_func("one_episode")
        # swap two agents, and re-assign camp
        self.agents.reverse()
        # first_frame_no = req_pb.frame_no
        # state : game_id / hero_runtime_id(camp) / frame_no /
        #
        # game_id = state_dict_list[0]['game_id']

        # swap two agents, and re-assign camp
        # self.agents.reverse()
        for i, agent in enumerate(self.agents):
            LOG.debug("reset agent {}".format(i))
            # camp = agent_camp[i]
            if eval:
                if load_models is None:
                    agent.reset("common_ai")
                else:
                    if load_models[i] is None:
                        agent.reset("common_ai")
                    else:
                        agent.reset("network", model_path=load_models[i])
            else:
                agent.reset(Config.ENEMY_TYPE)

        # restart a new game
        LOG.debug("reset env")
        camp_game_id, is_gameover = self.env.reset(self.agents, eval_mode=eval)
        first_time = True
        first_frame_no = -1

        # reset mem pool and models
        LOG.debug("reset sample_manager")
        sample_manager.reset(agents=self.agents, game_id=camp_game_id)
        log_time_func("reset", end=True)
        game_info = {}

        while not done:
            log_time_func("one_frame")

            if is_gameover:
                break
            pro_type = 0
            # while True:
            is_send = 0
            reward_camp = [[], []]
            all_hero_reward_camp = [[], []]
            for i, agent in enumerate(self.agents):
                if agent.is_common_ai():
                    LOG.info("agent %d is common_ai" % i)
                    continue
                pro_type, features, req_pb = self.env.step_feature(i)
                # print("frame_no: %d" %(req_pb.frame_no))
                if pro_type == 0:
                    continue
                if first_time:
                    first_frame_no = req_pb.frame_no
                    LOG.info("first_frame_no %d" % first_frame_no)
                    first_time = False

                for hero_idx in range(self.hero_num):
                    reward_camp[i].append(features[hero_idx].reward)
                    all_hero_reward_camp[i].append(
                        features[hero_idx].all_hero_reward_info
                    )

                sample = {}
                if not req_pb.gameover:
                    prob, lstm_info = agent.predict_process(features, req_pb)
                    sample = self.env.step_action(prob, features, req_pb, i, lstm_info)
                else:
                    self.env._gameover(i)

                is_send = req_pb.gameover or (
                    ((req_pb.frame_no - first_frame_no) % self.send_sample_frame == 0)
                    and (req_pb.frame_no > first_frame_no)
                )
                if agent.is_latest_model and not eval:
                    if not is_send:
                        sample_manager.save_sample(
                            **sample,
                            agent_id=i,
                            game_id=camp_game_id[i],
                            uuid=self.m_task_uuid
                        )
                    else:
                        LOG.info(
                            "save_last_sample frame[%d] req_pb.gameover[%d]"
                            % (
                                req_pb.frame_no,
                                req_pb.gameover,
                            )
                        )
                        if req_pb.gameover:
                            sample_manager.save_last_sample(
                                agent_id=i,
                                reward=reward_camp[i],
                                all_hero_reward_s=all_hero_reward_camp[i],
                            )
                        else:
                            sample_manager.save_last_sample(
                                agent_id=i,
                                reward=reward_camp[i],
                                value_s=sample["value_s"],
                                all_hero_reward_s=all_hero_reward_camp[i],
                            )
            if pro_type == 0:
                continue

            if is_send:
                if IS_TRAIN and not eval:
                    LOG.info("send_sample and update model")
                    sample_manager.send_samples()
                    sample_manager.reset(agents=self.agents, game_id=camp_game_id)
                    for i, agent in enumerate(self.agents):
                        agent.update_model()
                    LOG.info("send done.")
                if req_pb.gameover:
                    is_gameover = True
                    done = True
                    LOG.info("req_pb.gameover frame_no: %d" % (req_pb.frame_no))
        log_time_func("one_frame", end=True)
        self.env.close_game(self.agents)

        game_info["length"] = req_pb.frame_no
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
            if eval:
                agent_model = load_models[i]
                if agent_model == None:
                    agent_model = "common_ai"
                LOG.info(
                    "camp%d_model:%s win:%d" % (agent_camp, agent_model, agent_win)
                )
            else:
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

        log_time_func("one_episode", end=True)
        # print game information
        # self._print_info(game_id, game_info, episode_infos, eval, eval_info)

    def _check_gamecore_state(self, state_dict_list):
        all_key_right = True
        key_list = ["feature", "camp_id"]
        for key_name in key_list:
            for state_dict in state_dict_list:
                if key_name not in state_dict:
                    all_key_right = False
        return all_key_right

    def _print_info(self, game_id, game_info, episode_infos, eval, eval_info=""):
        # TODO: update this code, and add some details about reward.
        if eval and len(eval_info) > 0:
            LOG.info("eval_info: %s" % eval_info)
        LOG.info("=" * 50)
        LOG.info("game_id : %s" % game_id)
        for item in g_log_time.items():
            if len(item) <= 1 or len(item[1]) == 0 or len(item[0]) == 0:
                continue
            mean = np.mean(item[1])
            max = np.max(item[1])
            LOG.info(
                "%s | mean:%s max:%s times:%s" % (item[0], mean, max, len(item[1]))
            )
            g_log_time[item[0]] = []
        LOG.info("=" * 50)
        for i, agent in enumerate(self.agents):
            LOG.info(
                "Agent is_main:{}, type:{}, camp:{},reward:{:.3f}, win:{}, h_act_rate:{}".format(
                    agent.keep_latest and eval,
                    agent.agent_type,
                    agent.hero_camp,
                    episode_infos[i]["reward"],
                    episode_infos[i]["win"],
                    1.0,
                )
            )
            LOG.info(
                "Agent is_main:{}, money_pre_frame:{:.2f}, kill:{}, death:{}, hurt_pf:{:.2f}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["money_pre_frame"],
                    episode_infos[i]["kill"],
                    episode_infos[i]["death"],
                    episode_infos[i]["hurt_pre_frame"],
                )
            )
        LOG.info("game info length:{}".format(game_info["length"]))

        LOG.info("=" * 50)

    def run(self, eval_mode=False, eval_number=-1, load_models=[]):
        self._last_print_time = time.time()
        self._episode_num = 0
        MAX_REPEAT_ERR_NUM = 2
        repeat_num = MAX_REPEAT_ERR_NUM
        if eval_mode:
            LOG.info("eval_mode start...")
            agent_0, agent_1 = 0, 1
            cur_models = [load_models[agent_0], load_models[agent_1]]
            cur_eval_cnt = 1
            swap = False

        while True:
            try:
                # provide a init eval value at the first episode
                if eval_mode:
                    if swap:
                        eval_info = "{} vs {}, {}/{}".format(
                            agent_1, agent_0, cur_eval_cnt, eval_number
                        )
                    else:
                        eval_info = "{} vs {}, {}/{}".format(
                            agent_0, agent_1, cur_eval_cnt, eval_number
                        )

                    LOG.info(cur_models)
                    LOG.info(eval_info)
                    self._run_episode(True, load_models=cur_models, eval_info=eval_info)
                    # swap camp
                    cur_models.reverse()
                    swap = not swap
                else:
                    # self._run_episode((self._episode_num+0) % Config.EVAL_FREQ == 0 and self.m_config_id == 0)
                    self._run_episode(
                        (self._episode_num + 0) % Config.EVAL_FREQ == 0
                        and self.m_config_id == -1
                    )
                self._episode_num += 1
                repeat_num = MAX_REPEAT_ERR_NUM
            except Exception as e:
                LOG.error(
                    "_run_episode err: {}/{}".format(repeat_num, MAX_REPEAT_ERR_NUM)
                )
                LOG.error(e)
                traceback.print_tb(e.__traceback__)
                repeat_num -= 1
                if repeat_num == 0:
                    raise e

            if eval_mode:
                # update eval agents and eval cnt
                cur_eval_cnt += 1
                if cur_eval_cnt > eval_number:
                    cur_eval_cnt = 1

                    agent_1 += 1
                    if agent_1 >= len(load_models):
                        agent_0 += 1
                        agent_1 = agent_0 + 1

                    if agent_1 >= len(load_models):
                        # eval end
                        break

                    cur_models = [load_models[agent_0], load_models[agent_1]]

            if 0 < self._max_episode <= self._episode_num:
                break
