# -*- coding: utf-8 -*-
import collections
import numpy as np
import random

from rl_framework.common.logging import log_time
from rl_framework.common.logging import logger as LOG
from rl_framework.mem_pool import MemPoolAPIs

from rl_data_info import RLDataInfo


class SampleManager:
    def __init__(
        self,
        mem_pool_addr,
        num_agents,
        single_test=False,
        sample_one_size=4921,
        lstm_unit_size=16,
        lstm_time_steps=16,
        gamma=0.995,
        lamda=0.95,
    ):
        self.single_test = single_test
        # connect to mem pool
        mem_pool_addr = mem_pool_addr
        ip, port = mem_pool_addr.split(":")
        self.m_mem_pool_ip = ip
        self.m_mem_pool_port = port

        if not self.single_test:
            self._mem_pool_api = MemPoolAPIs(
                self.m_mem_pool_ip,
                self.m_mem_pool_port,
                socket_type="zmq",
            )

        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]

        self.hero_num = 3

        self.gamma = gamma
        self.lamda = lamda
        self.lstm_time_steps = lstm_time_steps
        self.lstm_unit_size = lstm_unit_size
        self.sample_one_size = sample_one_size

    def reset(self):
        LOG.debug("reset sample_manager")
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    @log_time("save_sample")
    def save_sample(
        self,
        frame_no,
        vec_feature_s,
        legal_action_s,
        action_s,
        reward_s,
        value_s,
        prob_s,
        sub_action_s,
        lstm_cell,
        lstm_hidden,
        agent_id,
        is_train,
    ):
        """
        samples must saved by frame_no order
        """
        lstm_cell = lstm_cell.flatten()
        lstm_hidden = lstm_hidden.flatten()
        one_camp_data = []
        for hero_idx in range(self.hero_num):
            reward_s[hero_idx] = self._clip_reward(reward_s[hero_idx])
            rl_data_info = RLDataInfo()

            value = value_s[hero_idx]

            # update last frame's next_value
            if len(self.rl_data_map[agent_id]) > 0:
                last_key = list(self.rl_data_map[agent_id].keys())[-1]
                last_rl_data_info = self.rl_data_map[agent_id][last_key]
                last_rl_data_info[hero_idx].next_value = value_s[hero_idx]
                last_rl_data_info[hero_idx].reward = reward_s[hero_idx]

            # save current sample
            rl_data_info.frame_no = frame_no

            rl_data_info.feature = np.array(vec_feature_s[hero_idx]).reshape([-1])
            rl_data_info.legal_action = np.array(legal_action_s[hero_idx]).reshape([-1])
            rl_data_info.reward = 0
            rl_data_info.value = value

            rl_data_info.lstm_info = np.concatenate([lstm_cell, lstm_hidden]).reshape(
                [-1]
            )

            # np: (12 + 16 + 16 + 16 + 14)
            rl_data_info.prob = np.array(prob_s[hero_idx]).reshape([-1])
            # np: (6)
            # rl_data_info.action = 0 if action < 0 else action
            rl_data_info.action = action_s[hero_idx]
            # np: (6)
            rl_data_info.sub_action = sub_action_s[hero_idx]
            rl_data_info.is_train = (
                False if action_s[hero_idx][0] < 0 else is_train[hero_idx]
            )
            one_camp_data.append(rl_data_info)
        self.rl_data_map[agent_id][frame_no] = one_camp_data

    def save_last_sample(self, agent_id, reward, value_s=None):
        value_s = value_s or [0.0] * self.hero_num
        if len(self.rl_data_map[agent_id]) > 0:
            # TODO: is_action_executed, last_gamecore_act
            for hero_idx in range(self.hero_num):
                last_key = list(self.rl_data_map[agent_id].keys())[-1]
                last_rl_data_info = self.rl_data_map[agent_id][last_key]
                last_rl_data_info[hero_idx].next_value = value_s[hero_idx]
                last_rl_data_info[hero_idx].reward = reward[hero_idx]

    def send_samples(self):
        self._calc_reward()
        self._format_data()
        self._send_game_data()

    def _calc_reward(self):
        """
        Calculate cumulated reward and advantage with GAE.
        reward_sum: used for value loss
        advantage: used for policy loss
        V(s) here is a approximation of target network
        """
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae = [0.0] * self.hero_num
            LOG.info("calc_reward i:%d reversed_keys_size:%d" % (i, len(reversed_keys)))
            index = 0
            for j in reversed_keys:
                index += 1
                rl_info = self.rl_data_map[i][j]
                for hero_idx in range(self.hero_num):
                    delta = (
                        -rl_info[hero_idx].value
                        + rl_info[hero_idx].reward
                        + self.gamma * rl_info[hero_idx].next_value
                    )
                    gae[hero_idx] = gae[hero_idx] * self.gamma * self.lamda + delta
                    rl_info[hero_idx].advantage = gae[hero_idx]
                    rl_info[hero_idx].reward_sum = (
                        gae[hero_idx] + rl_info[hero_idx].value
                    )

                    LOG.debug(
                        "agent[%d] frame[%d] hero[%d] value[%f] next_value[%f] reward[%f] advantage[%f] reward_sum[%f]"
                        % (
                            i,
                            j,
                            hero_idx,
                            rl_info[hero_idx].value,
                            rl_info[hero_idx].next_value,
                            rl_info[hero_idx].reward,
                            rl_info[hero_idx].advantage,
                            rl_info[hero_idx].reward_sum,
                        )
                    )

    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros(
            [
                (np.prod(sample_batch[0].shape) + np.prod(sample_lstm[0].shape))
                * self.hero_num
            ]
        )
        # data :(16, data_size)   lstm: (unit_size*2, )
        sample_one_size = sample_batch[0].shape[1]
        sample_one_lstm = sample_lstm[0].shape[0]
        s_idx = 0
        e_idx = 0
        for hero_idx in range(self.hero_num):
            # hero_data
            # for frame in range(self._LSTM_FRAME):
            for frame in range(self.lstm_time_steps):
                s_idx = e_idx
                e_idx = s_idx + sample_one_size
                sample[s_idx:e_idx] = sample_batch[hero_idx][frame]
            # lstm data
            s_idx = e_idx
            e_idx = s_idx + sample_one_lstm
            sample[s_idx:e_idx] = sample_lstm[hero_idx]
        return sample

    @log_time("sample_manger_format_data")
    def _format_data(self):
        # sample_one_size = np.sum(self._data_shapes[:-2])//self._LSTM_FRAME
        # sample_one_size = np.sum(self._data_shapes[0])
        sample_one_size = self.sample_one_size
        # unit_num * 2 (cell + hidden)
        # sample_lstm_size =  self._LSTM_UNIT_SIZE * 2
        sample_lstm_size = self.lstm_unit_size * 2
        # sample_batch = np.zeros([self.hero_num, self._LSTM_FRAME, sample_one_size])
        sample_batch = np.zeros([self.hero_num, self.lstm_time_steps, sample_one_size])
        sample_lstm = np.zeros([self.hero_num, sample_lstm_size])
        first_frame_no = -1

        for i in range(self.num_agents):
            cnt = 0
            index = 0
            frame_num = len(self.rl_data_map[i])
            for j in self.rl_data_map[i]:
                index += 1

                if (cnt > 0) and ((frame_num - index + 1) == self.lstm_time_steps):
                    LOG.info("reset cnt frame_num:%d index:%d" % (frame_num, index))
                    cnt = 0

                rl_info = self.rl_data_map[i][j]
                for hero_idx in range(self.hero_num):
                    if cnt == 0:
                        # lstm cell & hidden
                        first_frame_no = rl_info[hero_idx].frame_no

                    # serilize one frames
                    idx, dlen = 0, 0
                    # vec_data
                    dlen = rl_info[hero_idx].feature.shape[0]
                    sample_batch[hero_idx][cnt, idx : idx + dlen] = rl_info[
                        hero_idx
                    ].feature
                    idx += dlen

                    # legal_action
                    dlen = rl_info[hero_idx].legal_action.shape[0]
                    sample_batch[hero_idx][cnt, idx : idx + dlen] = rl_info[
                        hero_idx
                    ].legal_action.flatten()
                    idx += dlen

                    # reward_sum & advantage
                    # LOG.error("reward_sum {}, {}".format(rl_info.reward_sum, type(rl_info.reward_sum)))
                    sample_batch[hero_idx][cnt, idx] = rl_info[hero_idx].reward_sum
                    idx += 1
                    sample_batch[hero_idx][cnt, idx] = rl_info[hero_idx].advantage
                    idx += 1

                    # labels
                    dlen = 5
                    sample_batch[hero_idx][cnt, idx : idx + dlen] = rl_info[
                        hero_idx
                    ].action
                    idx += dlen

                    # probs (neg log pi->prob)
                    dlen = rl_info[hero_idx].prob.shape[0]
                    sample_batch[hero_idx][cnt, idx : idx + dlen] = rl_info[
                        hero_idx
                    ].prob.flatten()
                    idx += dlen
                    # for p in rl_info[hero_idx].prob:
                    #    dlen = len(p)
                    #    # p = np.exp(-nlp)
                    #    # p = p / np.sum(p)
                    #    sample_batch[hero_idx][cnt, idx:idx + dlen] = p
                    #    idx += dlen

                    # is_train
                    sample_batch[hero_idx][cnt, idx] = rl_info[hero_idx].is_train
                    idx += 1

                    # sub_action
                    dlen = 5
                    sample_batch[hero_idx][cnt, idx : idx + dlen] = rl_info[
                        hero_idx
                    ].sub_action
                    idx += dlen

                    LOG.debug(
                        "format_data frame_no:%d hero_idx:%d is_train:%d sub_action:%s action:%s"
                        % (
                            rl_info[hero_idx].frame_no,
                            hero_idx,
                            rl_info[hero_idx].is_train,
                            str(rl_info[hero_idx].sub_action),
                            str(rl_info[hero_idx].action),
                        )
                    )

                assert idx == sample_one_size, "Sample check failed, {}/{}".format(
                    idx, sample_one_size
                )

                cnt += 1
                if cnt == self.lstm_time_steps:
                    # LOG.info("format_data i:%d j:%d cnt:%d first_frame_no:%d index:%d frame_num:%d" % (i, j, cnt, first_frame_no, index, frame_num))
                    cnt = 0
                    # reshape sample batch and put into sample
                    sample = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                    self.m_replay_buffer[i].append((first_frame_no, sample))
                    for hero_idx in range(self.hero_num):
                        sample_lstm[hero_idx] = rl_info[hero_idx].lstm_info

    def _clip_reward(self, reward, max=100, min=-100):
        if reward > max:
            reward = max
        elif reward < min:
            reward = min
        return reward

    def _add_extra_info(self, frame_no, sample):
        return sample.astype(np.float32).tobytes()

    def _send_game_data(self):
        for i in range(self.num_agents):
            # if i == 0:
            #    continue
            samples = []
            for sample in self.m_replay_buffer[i]:
                samples.append(self._add_extra_info(*sample))
            if (not self.single_test) and len(samples) > 0:
                LOG.info("SendSample agent_id:%d sample_size:%d" % (i, len(samples)))
                self._mem_pool_api.push_samples(samples)
