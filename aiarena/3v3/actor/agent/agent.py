import time
from collections import defaultdict
import numpy as np
import random

from rl_framework.predictor.utils import (
    cvt_tensor_to_infer_input,
    cvt_tensor_to_infer_output,
)
from rl_framework.model_pool import ModelPoolAPIs

from hok.common.log import log_time
from hok.common.log import logger as LOG

_g_rand_max = 10000
_g_model_update_ratio = 0.8


def cvt_infer_list_to_numpy_list(infer_list):
    data_list = [infer.data for infer in infer_list]
    return data_list


class Agent:
    def __init__(
        self,
        model,
        model_pool_addr,
        keep_latest=False,
        rule_only=False,
        backend="tensorflow",
        single_test=False,
    ):
        self.model = model
        self.backend = backend
        self.rule_only = rule_only
        self.single_test = single_test
        self.agent_type = "network"

        self.is_latest_model: bool = False
        self.keep_latest = keep_latest
        if self.rule_only:
            return

        LOG.info(model_pool_addr)
        if self.backend == "tensorflow":
            from rl_framework.predictor.predictor.local_predictor import (
                LocalCkptPredictor as LocalPredictor,
            )

            self._predictor = LocalPredictor(self.model.build_infer_graph())
        else:
            from rl_framework.predictor.predictor.local_torch_predictor import (
                LocalTorchPredictor,
            )

            self._predictor = LocalTorchPredictor(self.model)

        if not model_pool_addr:
            self._model_pool_api = None
        else:
            self._model_pool_api = ModelPoolAPIs(model_pool_addr)

        self.model_version = ""
        self.model_list = []

        self.lstm_unit_size = self.model.lstm_unit_size

        self.lstm_hidden = None
        self.lstm_cell = None

        self.last_model_path = None

    def reset(self, agent_type=None, model_path=None):
        if self.rule_only:
            return

        # reset lstm input
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])

        if agent_type is not None:
            if self.keep_latest:
                self.agent_type = "network"
            else:
                self.agent_type = agent_type

        # for test without model pool
        if self.single_test:
            self.is_latest_model = True
            if self.backend == "tensorflow":
                LOG.info("SINGLE_TEST: backend=tensorflow")
                self._predictor._sess.run(self.model.init)
            elif self.backend == "pytorch":
                LOG.info("SINGLE_TEST: backend=pytorch")
            else:
                raise NotImplementedError(
                    "SINGLE_TEST: backend not in [tensorflow, pytorch]..."
                )
        else:
            if model_path is None:
                while True:
                    try:
                        if self.keep_latest:
                            self._get_latest_model()
                        else:
                            self._get_random_model()
                        return
                    except Exception as e:
                        LOG.error(e)
                        LOG.error("get_model error, try again...")
                        time.sleep(1)
            elif self.rule_only:
                self._get_random_model()
            else:
                ret = self._predictor.load_model(model_path)

    def is_common_ai(self):
        if self.agent_type == "common_ai":
            return True
        else:
            return False

    def _update_model_list(self):
        model_key_list = []
        while len(model_key_list) == 0:
            model_key_list = self._model_pool_api.pull_keys()
            if not model_key_list:
                LOG.warning("No model in model_pool, wait for 1 sec...")
                time.sleep(1)
        self.model_list = model_key_list

    def _load_model(self, model_version):
        if model_version == self.model_version:
            return True
        model_path = self._model_pool_api.pull_model_path(model_version)
        model_path = "%s/checkpoint" % (model_path)
        LOG.info("load model: {} in {}".format(model_version, model_path))
        ret = self._predictor.load_model(model_path)
        if ret:
            # if failed, do not update model_version
            self.model_version = model_version
        return ret

    def _get_random_model(self):
        if self.agent_type in ["common_ai", "random"] or self.rule_only:
            self.is_latest_model = False
            if self.backend == "tensorflow":
                self._predictor._sess.run(self.model.init)
            elif self.backend == "pytorch":
                pass
            else:
                raise NotImplementedError(
                    "_get_random_model: backend not in [tensorflow, pytorch]..."
                )
            self.model_version = ""
            return True

        self._update_model_list()
        rand_float = float(random.uniform(0, _g_rand_max)) / float(_g_rand_max)
        if rand_float <= _g_model_update_ratio:
            midx = len(self.model_list) - 1
            self.is_latest_model = True
        else:
            midx = int(random.random() * len(self.model_list))
            if midx == len(self.model_list):
                midx = len(self.model_list) - 1
            self.is_latest_model = False
        return self._load_model(self.model_list[midx])

    def _get_latest_model(self):
        self._update_model_list()
        self.is_latest_model = True
        return self._load_model(self.model_list[-1])

    def update_model(self):
        if not self.single_test and self.is_latest_model:
            self._get_latest_model()
            LOG.info("latest model, update")
        else:
            LOG.info("old model, not update")

    # return action, sample, reward, done
    @log_time("aiprocess_process")
    def predict_process(self, features, frame_state):
        runtime_ids = []
        for hero_idx in range(len(features)):
            runtime_ids.append(features[hero_idx].hero_runtime_id)

        if self.backend == "tensorflow":
            pred_ret, lstm_info = self._predict_process(
                features, frame_state, runtime_ids
            )
        else:
            pred_ret, lstm_info = self._predict_process_torch(
                features, frame_state, runtime_ids
            )

        return pred_ret, lstm_info

    def _predict_process(self, features, frame_state, runtime_ids):
        # put data to input
        input_list = cvt_tensor_to_infer_input(self.model.get_input_tensors())
        for i in range(len(features)):
            input_list[i].set_data(np.array(features[i].feature))
        input_list[len(features)].set_data(self.lstm_cell)
        input_list[len(features) + 1].set_data(self.lstm_hidden)

        output_list = cvt_tensor_to_infer_output(self.model.get_output_tensors())
        output_list = self._predictor.inference(
            input_list=input_list, output_list=output_list
        )
        # cvt output data
        np_output = cvt_infer_list_to_numpy_list(output_list)

        prob = np_output[: len(features)]
        lstm_info = np_output[-2:]
        self.lstm_cell, self.lstm_hidden = lstm_info

        return prob, lstm_info

    def _predict_process_torch(self, features, frame_state, runtime_ids):
        # put data to input
        input_list = []
        for i in range(len(features)):
            input_list.append(np.array(features[i].feature))
        input_list.append(self.lstm_cell)
        input_list.append(self.lstm_hidden)

        output_list = self._predictor.inference(input_list)
        np_output_list = []
        for output in output_list:
            np_output_list.append(output.numpy())

        prob = np_output_list[: len(features)]
        lstm_info = np_output_list[-2:]
        self.lstm_cell, self.lstm_hidden = lstm_info

        return prob, lstm_info

    def compute_reward(self, features, frame_state):
        hero_rid_rewards = features[0].hero_rid_reward

        team_rewards = defaultdict(list)
        hero_reward = defaultdict(float)
        for hero in frame_state.hero_list:
            rid = hero.runtime_id
            reward = sum(hero_rid_rewards[rid].values())
            hero_reward[rid] = reward
            team_rewards[hero.camp].append(reward)

        team_reward_avg = defaultdict(float)
        for camp, hero_rewards in team_rewards.items():
            team_reward_avg[camp] = sum(hero_rewards) / len(hero_rewards)

        m_whether_use_zero_sum_reward = 1
        m_team_spirit = 0
        m_time_scaling_discount = 1
        m_time_scaling_time = 4500

        rewards = []
        for feature in features:
            rid = feature.hero_runtime_id
            camp_id = feature.camp_id

            main_hero_reward = hero_reward[rid]

            if camp_id == 1:
                ally_camp_mean_reward = team_reward_avg[1]
                enemy_camp_mean_reward = team_reward_avg[2]
            elif camp_id == 2:
                ally_camp_mean_reward = team_reward_avg[2]
                enemy_camp_mean_reward = team_reward_avg[1]
            else:
                ally_camp_mean_reward = 0
                enemy_camp_mean_reward = 0
                LOG.warning("Unknow hero camp: {}", camp_id)

            main_hero_reward_final = (
                (1 - m_team_spirit) * main_hero_reward
                + m_team_spirit * ally_camp_mean_reward
                - m_whether_use_zero_sum_reward * enemy_camp_mean_reward
            )
            main_hero_reward_final *= pow(
                m_time_scaling_discount, (frame_state.frame_no / m_time_scaling_time)
            )
            rewards.append(main_hero_reward_final)

            # assert main_hero_reward_final - feature.reward < 1e-5

        return rewards

    def sample_process(self, features, results, lstm_info, frame_state):
        # 自定义reward示例
        # rewards = self.compute_reward(features, frame_state)

        lstm_cell, lstm_hidden = lstm_info

        # probs : two camp , 3 heros, label prob +  value
        frame_no = frame_state.frame_no

        feature_one_camp = []
        reward_one_camp = []
        actions_one_camp = []
        sub_actions_one_camp = []
        legal_action_one_camp = []
        prob_one_camp = []
        value_one_camp = []
        is_train_one_camp = []
        for hero in range(len(features)):
            feature_one_camp.append(np.array(features[hero].feature))
            legal_action_one_camp.append(sum(results[hero].legal_action, []))
            actions_one_camp.append(results[hero].actions)
            reward_one_camp.append(features[hero].reward)
            tmp_prob = sum(results[hero].final_prob_list, [])
            prob_one_camp.append(tmp_prob[:-1])
            value_one_camp.append(tmp_prob[-1])
            sub_actions_one_camp.append(results[hero].sub_actions)
            is_train_one_camp.append(results[hero].is_train)

        sample = {
            "frame_no": frame_no,
            "vec_feature_s": feature_one_camp,
            "legal_action_s": legal_action_one_camp,
            "action_s": actions_one_camp,
            "reward_s": reward_one_camp,
            "value_s": value_one_camp,
            "prob_s": prob_one_camp,
            "sub_action_s": sub_actions_one_camp,
            "lstm_cell": lstm_cell,
            "lstm_hidden": lstm_hidden,
            "is_train": is_train_one_camp,
        }
        return sample
