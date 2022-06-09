# -*- coding: utf-8 -*-
import collections
import random

import numpy as np
from common_config import Config, ModelConfig
from framework.common.common_func import *
from framework.common.common_log import CommonLogger
from framework.common.rl_data_info import RLDataInfo
from rl_framework.mem_pool import MemPoolAPIs

LOG = CommonLogger.get_logger()


class SampleManager:
    def __init__(
        self, mem_pool_addr, mem_pool_type, num_agents, game_id=None, local_mode=False
    ):
        # connect to mem pool
        # deal with multiple mem_pool, randomly select one to connect!
        mem_pool_addr = mem_pool_addr.strip().split(";")
        LOG.info("mempool list: {}".format(mem_pool_addr))
        idx = random.randint(0, len(mem_pool_addr) - 1)
        mem_pool_addr = mem_pool_addr[idx]
        LOG.info("connect to mempool: {}".format(mem_pool_addr))
        ip, port = mem_pool_addr.split(":")
        self.m_mem_pool_ip = ip
        self.m_mem_pool_port = port
        print("Use mem_pool_addr", mem_pool_addr)
        self._data_shapes = ModelConfig.data_shapes
        self._LSTM_FRAME = ModelConfig.LSTM_TIME_STEPS

        if Config.SINGLE_TEST or local_mode:
            mem_pool_type = "zmq"
        else:
            self._mem_pool_api = MemPoolAPIs(
                self.m_mem_pool_ip, self.m_mem_pool_port, socket_type=mem_pool_type
            )

        self.m_game_id = game_id
        self.m_task_id, self.m_task_uuid = 0, "default_task_uuid"
        self.num_agents = num_agents
        self.agents = None
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]

        # load config from config file
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA

        # self.sample_parse_lib = interface.SampleParse()

    def reset(self, agents, game_id):
        self.m_game_id = game_id
        self.agents = agents
        self.num_agents = len(agents)
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    @log_time("save_sample")
    def save_sample(
        self,
        frame_no,
        vec_feature,
        legal_action,
        action,
        reward,
        value,
        prob,
        sub_action,
        lstm_cell,
        lstm_hidden,
        done,
        agent_id,
        is_train=True,
        game_id=None,
        uuid=None,
    ):
        """
        samples must saved by frame_no order
        """
        reward = self._clip_reward(reward)
        rl_data_info = RLDataInfo()
        # rl_data_info.game_id = struct.pack('%ss' % len(game_id), bytes(game_id, encoding='utf8'))

        value = value.flatten()[0]
        lstm_cell = lstm_cell.flatten()
        lstm_hidden = lstm_hidden.flatten()

        # if frame_no in self.rl_data_map[agent_id].keys():
        #     LOG.error("Error: repeated frame {} in sample manager {}.".format(frame_no, agent_id))
        #     assert False

        # update last frame's next_value
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = value
            last_rl_data_info.reward = reward

        # save current sample
        rl_data_info.frame_no = frame_no

        rl_data_info.feature = vec_feature.reshape([-1])
        rl_data_info.legal_action = legal_action.reshape([-1])
        rl_data_info.reward = 0
        rl_data_info.value = value
        # rl_data_info.done = done

        rl_data_info.lstm_info = np.concatenate([lstm_cell, lstm_hidden]).reshape([-1])

        # np: (12 + 16 + 16 + 16 + 14)
        rl_data_info.prob = prob
        # np: (6)
        # rl_data_info.action = 0 if action < 0 else action
        rl_data_info.action = action
        # np: (6)
        rl_data_info.sub_action = sub_action[action[0]]
        rl_data_info.is_train = False if action[0] < 0 else is_train

        # LOG.error("save {}, {}, {}".format(rl_data_info.value, rl_data_info.reward, rl_data_info.next_value))

        # rl_data_info.task_uuid = struct.pack('%ss' % len(uuid), bytes(uuid, encoding="utf8"))
        self.rl_data_map[agent_id][frame_no] = rl_data_info

    def save_last_sample(self, reward, agent_id):
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = 0
            last_rl_data_info.reward = reward

    # def save_value(self, value):
    #     str_q_value = value.tostring()
    #     self.rl_data_map[sorted(self.rl_data_map.keys())[-1]].next_Q_value = str_q_value

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
            gae, last_gae = 0.0, 0.0
            for j in reversed_keys:
                rl_info = self.rl_data_map[i][j]
                delta = (
                    -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                )
                gae = gae * self.gamma * self.lamda + delta
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value

    # data_keys = "vec_data,reward,advantage,label0,label1,label2,label3,label4,label5,prob0,prob1,prob2,prob3,prob4," \
    #             "prob5,weight0,weight1,weight2,weight3,weight4,weight5,is_train, lstm_cell, lstm_hidden_state"
    # _LSTM_FRAME = 16

    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)])
        idx, s_idx = 0, 0

        sample[-sample_lstm.shape[0] :] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx : s_idx + split_shape[0]] = sample_batch[
                :, idx : idx + one_shape
            ].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]
        return sample

    @log_time("sample_manger_format_data")
    def _format_data(self):
        sample_one_size = np.sum(self._data_shapes[:-2]) // self._LSTM_FRAME
        sample_lstm_size = np.sum(self._data_shapes[-2:])
        sample_batch = np.zeros([self._LSTM_FRAME, sample_one_size])
        sample_lstm = np.zeros([sample_lstm_size])
        first_frame_no = -1

        for i in range(self.num_agents):
            cnt = 0
            for j in self.rl_data_map[i]:
                rl_info = self.rl_data_map[i][j]
                if cnt == 0:
                    # lstm cell & hidden
                    first_frame_no = rl_info.frame_no
                    sample_lstm = rl_info.lstm_info

                # serilize one frames
                idx, dlen = 0, 0
                # vec_data
                dlen = rl_info.feature.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.feature
                idx += dlen

                # legal_action
                dlen = rl_info.legal_action.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.legal_action
                idx += dlen

                # reward_sum & advantage
                # LOG.error("reward_sum {}, {}".format(rl_info.reward_sum, type(rl_info.reward_sum)))
                sample_batch[cnt, idx] = rl_info.reward_sum
                idx += 1
                sample_batch[cnt, idx] = rl_info.advantage
                idx += 1

                # labels
                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.action
                idx += dlen

                # probs (neg log pi->prob)
                for p in rl_info.prob:
                    dlen = len(p)
                    # p = np.exp(-nlp)
                    # p = p / np.sum(p)
                    sample_batch[cnt, idx : idx + dlen] = p
                    idx += dlen

                # sub_action
                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.sub_action
                idx += dlen

                # is_train
                sample_batch[cnt, idx] = rl_info.is_train
                idx += 1

                assert idx == sample_one_size, "Sample check failed, {}/{}".format(
                    idx, sample_one_size
                )

                cnt += 1
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    # reshape sample batch and put into sample
                    # sample = np.zeros([np.sum(self._data_shapes)])
                    sample = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                    self.m_replay_buffer[i].append((first_frame_no, sample))

    def _clip_reward(self, reward, max=100, min=-100):
        if reward > max:
            reward = max
        elif reward < min:
            reward = min
        return reward

    # send game info like: ((size, data))*5:
    def _add_extra_info(self, frame_no, sample):
        return sample.astype(np.float32).tobytes()

    def _send_game_data(self):
        for i in range(self.num_agents):
            samples = []
            for sample in self.m_replay_buffer[i]:
                samples.append(self._add_extra_info(*sample))
            if (not Config.SINGLE_TEST) and len(samples) > 0:
                self._mem_pool_api.push_samples(samples)
