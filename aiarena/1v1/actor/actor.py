# -*- coding: utf-8 -*-
"""
    KingHonour Data production process
"""
import os
import time
import traceback

import numpy as np
from rl_framework.common.logging import log_time_func, g_log_time
import rl_framework.common.logging as LOG


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
OS_ENV = os.environ
IS_DEV = OS_ENV.get("IS_DEV")


class Actor:
    """
    used for sample logic
        run 1 episode
        save sample in sample manager
    """

    # def __init__(self, id, type):
    def __init__(
        self,
        id,
        agents,
        max_episode: int = 0,
        env=None,
        monitor_logger=None,
        camp_iter=None,
        is_train=True,
        enemy_type="network",
    ):
        self.m_config_id = id
        self.m_task_uuid = "TODO TASK_UUID"
        self.env = env
        self._max_episode = max_episode
        self._episode_num = 0
        self.agents = agents
        self.monitor_logger = monitor_logger
        self.camp_iter = camp_iter
        self.is_train = is_train
        self.enemy_type = enemy_type

    def upload_monitor_data(self, data: dict):
        if self.monitor_logger:
            self.monitor_logger.info(data)

    def set_env(self, environment):
        self.env = environment

    def set_sample_manager(self, sample_manager):
        self.m_sample_manager = sample_manager

    def set_agents(self, agents):
        self.agents = agents

    def _get_common_ai(self, eval, load_models):
        use_common_ai = [False] * len(self.agents)
        for i, agent in enumerate(self.agents):
            if eval:
                if load_models is None or len(load_models) < 2:
                    if not agent.keep_latest:
                        use_common_ai[i] = True
                elif load_models[i] is None:
                    use_common_ai[i] = True

        return use_common_ai

    def _reload_agents(self, eval=False, load_models=None):
        for i, agent in enumerate(self.agents):
            LOG.debug("reset agent {}".format(i))
            if eval:
                if load_models is None or len(load_models) < 2:
                    agent.reset("common_ai")
                else:
                    if load_models[i] is None:
                        agent.reset("common_ai")
                    else:
                        agent.reset("network", model_path=load_models[i])
            else:
                if len(load_models) == 1 and not agent.keep_latest:
                    agent.reset(self.enemy_type, model_path=load_models[0])
                else:
                    agent.reset(self.enemy_type)

    def _save_last_sample(self, done, eval, sample_manager, state_dict):
        if done:
            for i, agent in enumerate(self.agents):
                if agent.is_latest_model and not eval:
                    if state_dict[i]["reward"] is not None:
                        if type(state_dict[i]["reward"]) == tuple:
                            # if reward is a vec
                            sample_manager.save_last_sample(
                                agent_id=i, reward=state_dict[i]["reward"][-1]
                            )
                        else:
                            # if reward is a float number
                            sample_manager.save_last_sample(
                                agent_id=i, reward=state_dict[i]["reward"]
                            )
                    else:
                        sample_manager.save_last_sample(agent_id=i, reward=0)

    def _run_episode(self, camp_config, eval=False, load_models=None):
        LOG.info("Start a new game")
        for item in g_log_time.items():
            g_log_time[item[0]] = []
        sample_manager = self.m_sample_manager
        done = False
        log_time_func("reset")
        log_time_func("one_episode")
        LOG.debug("reset env")
        LOG.info(camp_config)
        use_common_ai = self._get_common_ai(eval, load_models)

        # ATTENTION: agent.reset() loads models from local file which cost a lot of time.
        #            Before upload your code, please check your code to avoid ANY time-wasting
        #            operations between env.reset() and env.close_game(). Any TIMEOUT in a round
        #            of game will cause undefined errors.

        # reload agent models
        self._reload_agents(eval, load_models)
        # restart a new game
        # reward :[dead,ep_rate,exp,hp_point,kill,last_hit,money,tower_hp_point,reward_sum]
        _, r, d, state_dict = self.env.reset(
            camp_config, use_common_ai=use_common_ai, eval=eval
        )
        if state_dict[0] is None:
            game_id = state_dict[1]["game_id"]
        else:
            game_id = state_dict[0]["game_id"]

        # update agents' game information
        for i, agent in enumerate(self.agents):
            player_id = self.env.player_list[i]
            camp = self.env.player_camp.get(player_id)
            agent.set_game_info(camp, player_id)

        # reset mem pool and models
        LOG.debug("reset sample_manager")
        sample_manager.reset(agents=self.agents, game_id=game_id)
        rewards = [[], []]
        step = 0
        log_time_func("reset", end=True)
        game_info = {}
        episode_infos = [{"h_act_num": 0} for _ in self.agents]

        while not done:
            log_time_func("one_frame")
            # while True:
            actions = []
            log_time_func("agent_process")
            for i, agent in enumerate(self.agents):
                if use_common_ai[i]:
                    actions.append(None)
                    rewards[i].append(0.0)
                    continue
                # print("agent{}".format(i),state_dict[i]['observation'])
                action, d_action, sample = agent.process(state_dict[i])
                if eval:
                    action = d_action
                # print("input act: [{}], {}, {}".format(i, action, state_dict[i]["legal_action"][:12]))
                actions.append(action)
                if action[0] == 10:
                    episode_infos[i]["h_act_num"] += 1
                rewards[i].append(sample["reward"])

                if agent.is_latest_model and not eval:
                    sample_manager.save_sample(
                        **sample, agent_id=i, game_id=game_id, uuid=self.m_task_uuid
                    )
            log_time_func("agent_process", end=True)

            log_time_func("step")
            # reward :[dead,ep_rate,exp,hp_point,kill,last_hit,money,tower_hp_point,reward_sum]
            _, r, d, state_dict = self.env.step(actions)
            # if np.isnan(r[0][-1]) or np.isnan(r[1][-1]):
            #     exit(0)
            log_time_func("step", end=True)

            req_pbs = self.env.cur_req_pb
            if req_pbs[0] is None:
                req_pb = req_pbs[1]
            else:
                req_pb = req_pbs[0]
            LOG.debug(
                "step: {}, frame_no: {}, reward: {}, {}".format(
                    step, req_pb.frame_no, r[0], r[1]
                )
            )
            step += 1
            done = d[0] or d[1]

            self._save_last_sample(done, eval, sample_manager, state_dict)
            log_time_func("one_frame", end=True)

        self.env.close_game()

        game_info["length"] = req_pb.frame_no
        loss_camp = -1
        camp_hp = {}
        all_camp_list = []
        for organ in req_pb.organ_list:
            if organ.type == 24:
                if organ.hp <= 0:
                    loss_camp = organ.camp
                camp_hp[organ.camp] = organ.hp
                all_camp_list.append(organ.camp)
            if organ.type in [21, 24]:
                LOG.info(
                    "Tower {} in camp {}, hp: {}".format(
                        organ.type, organ.camp, organ.hp
                    )
                )

        for i, agent in enumerate(self.agents):
            if use_common_ai[i]:
                continue
            for hero_state in req_pbs[i].hero_list:
                if agent.player_id == hero_state.runtime_id:
                    episode_infos[i]["money_per_frame"] = (
                        hero_state.moneyCnt / game_info["length"]
                    )
                    episode_infos[i]["kill"] = hero_state.killCnt
                    episode_infos[i]["death"] = hero_state.deadCnt
                    episode_infos[i]["assistCnt"] = hero_state.assistCnt
                    episode_infos[i]["hurt_per_frame"] = (
                        hero_state.totalHurt / game_info["length"]
                    )
                    episode_infos[i]["hurtH_per_frame"] = (
                        hero_state.totalHurtToHero / game_info["length"]
                    )
                    episode_infos[i]["hurtBH_per_frame"] = (
                        hero_state.totalBeHurtByHero / game_info["length"]
                    )
                    episode_infos[i]["heroes"] = camp_config["heroes"][i]
                    episode_infos[i]["totalHurtToHero"] = hero_state.totalHurtToHero
                    break
            if loss_camp == -1:
                episode_infos[i]["win"] = 0
            else:
                episode_infos[i]["win"] = -1 if agent.hero_camp == loss_camp else 1

            episode_infos[i]["reward"] = np.sum(rewards[i])
            episode_infos[i]["h_act_rate"] = episode_infos[i]["h_act_num"] / step

        if self.is_train and not eval:
            LOG.debug("send sample_manager")
            sample_manager.send_samples()
            LOG.debug("send done.")

        log_time_func("one_episode", end=True)
        # print game information
        self._print_info(
            game_id, game_info, episode_infos, eval, common_ai=use_common_ai
        )

    def _print_info(self, game_id, game_info, episode_infos, eval, common_ai=None):
        if common_ai is None:
            common_ai = [False] * len(self.agents)
        LOG.info("=" * 50)
        LOG.info("game_id : %s" % game_id)
        for item in g_log_time.items():
            if len(item) <= 1 or len(item[1]) == 0 or len(item[0]) == 0:
                continue
            mean = np.mean(item[1])
            max = np.max(item[1])
            sum = np.sum(item[1])
            LOG.info(
                "%s | sum: %s mean:%s max:%s times:%s"
                % (item[0], sum, mean, max, len(item[1]))
            )
            g_log_time[item[0]] = []
        LOG.info("=" * 50)
        for i, agent in enumerate(self.agents):
            if common_ai[i]:
                continue
            LOG.info(
                "Agent is_main:{}, type:{}, camp:{},reward:{:.3f}, win:{}, h_act_rate:{}, heroes:{}".format(
                    agent.keep_latest and eval,
                    agent.agent_type,
                    agent.hero_camp,
                    episode_infos[i]["reward"],
                    episode_infos[i]["win"],
                    episode_infos[i]["h_act_rate"],
                    episode_infos[i]["heroes"],
                )
            )
            LOG.info(
                "Agent is_main:{}, money_per_frame:{:.2f}, kill:{}, death:{}, hurt_pf:{:.2f}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["money_per_frame"],
                    episode_infos[i]["kill"],
                    episode_infos[i]["death"],
                    episode_infos[i]["hurt_per_frame"],
                )
            )
            if agent.keep_latest and eval:
                self.upload_monitor_data(
                    {
                        "reward": episode_infos[i]["reward"],
                        "win": episode_infos[i]["win"],
                        "hurt_per_frame": episode_infos[i]["hurt_per_frame"],
                        "money_per_frame": episode_infos[i]["money_per_frame"],
                        "totalHurtToHero": episode_infos[i]["totalHurtToHero"],
                        "kill": episode_infos[i]["kill"],
                        "death": episode_infos[i]["death"],
                        "assistCnt": episode_infos[i]["assistCnt"],
                        "hurtH_per_frame": episode_infos[i]["hurtH_per_frame"],
                        "hurtBH_per_frame": episode_infos[i]["hurtBH_per_frame"],
                        "win": episode_infos[i]["win"],
                        "length": game_info["length"],
                    }
                )
        LOG.info("game info length:{}".format(game_info["length"]))

        LOG.info("=" * 50)

    def run(self, load_models=None, eval_freq=5):

        self._episode_num = 0

        while True:
            camp_config = next(self.camp_iter)
            try:
                self._episode_num += 1
                # provide a init eval value at the first episode
                eval_with_common_ai = (
                    self.m_config_id == 0 and self._episode_num % eval_freq == 0
                )

                self._run_episode(
                    camp_config, eval_with_common_ai, load_models=load_models
                )
            except Exception as e:  # pylint: disable=broad-except
                LOG.error(e)
                traceback.print_exc()
                time.sleep(1)

            if 0 < self._max_episode <= self._episode_num:
                break

        for agent in self.agents:
            agent.close()
