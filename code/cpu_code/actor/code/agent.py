import random
import h5py
import numpy as np

from rl_framework.predictor.predictor.local_predictor import (
    LocalCkptPredictor as LocalPredictor,
)

from rl_framework.predictor.utils import (
    cvt_tensor_to_infer_input,
    cvt_tensor_to_infer_output,
)
from rl_framework.model_pool import ModelPoolAPIs

from framework.common.common_func import log_time
from config.config import ModelConfig, Config
import rl_framework.common.logging as LOG


_G_CHECK_POINT_PREFIX = "checkpoints_"
_G_RAND_MAX = 10000
_G_MODEL_UPDATE_RATIO = 0.8


def cvt_infer_list_to_numpy_list(infer_list):
    data_list = [infer.data for infer in infer_list]
    return data_list


class RandomAgent:
    def process(self, feature, legal_action):
        action = [random.randint(0, 2) - 1, random.randint(0, 2) - 1]
        value = [0.0]
        neg_log_pi = [0]
        return action, value, neg_log_pi


class Agent:
    def __init__(
        self,
        model_cls,
        model_pool_addr,
        keep_latest=False,
        local_mode=False,
        dataset=None,
    ):
        self.model = model_cls()
        self.graph = self.model.build_infer_graph()

        self._predictor = LocalPredictor(self.graph)
        if local_mode:
            self._model_pool_api = None
        else:
            self._model_pool_api = ModelPoolAPIs(model_pool_addr)

        self.model_version = ""
        self.is_latest_model: bool = False
        self.keep_latest = keep_latest
        self.model_list = []

        self.lstm_unit_size = ModelConfig.LSTM_UNIT_SIZE

        self.lstm_hidden = None
        self.lstm_cell = None

        # self.agent_type = "common_ai"
        self.player_id = 0
        self.hero_camp = 0
        self.last_model_path = None
        self.label_size_list = ModelConfig.LABEL_SIZE_LIST
        self.legal_action_size = ModelConfig.LEGAL_ACTION_SIZE_LIST

        # self.agent_type = "network"
        if self.keep_latest:
            self.agent_type = "network"
        else:
            self.agent_type = Config.ENEMY_TYPE

        if dataset is None:
            self.save_h5_sample = False
            self.dataset_name = None
            self.dataset = None
        else:
            self.save_h5_sample = True
            self.dataset_name = dataset
            self.dataset = h5py.File(dataset, "a")

    def set_game_info(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id

    # reset the agent,agent_type in ["network","common_ai"],if model_path is None,get model from model pool
    def reset(self, agent_type=None, model_path=None):
        # reset lstm input
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])

        if agent_type is not None:
            if self.keep_latest:
                self.agent_type = "network"
            else:
                self.agent_type = agent_type

        # for test without model pool
        if Config.SINGLE_TEST:
            self.is_latest_model = True
            self._predictor._sess.run(self.model.init)
        else:
            if model_path is None:
                while True:
                    try:
                        if self.keep_latest:
                            self._get_latest_model()
                        else:
                            self._get_random_model()
                        self.last_model_path = None
                        return
                    except Exception as e:  # pylint: disable=broad-except
                        LOG.error(e)
                        LOG.error("get_model error, try again...")
            else:
                if model_path != self.last_model_path:
                    self._predictor.load_model(model_path)
                    self.last_model_path = model_path
                else:
                    LOG.info(
                        "model {} alreadly load last time, skip now!".format(model_path)
                    )

        if self.dataset is None:
            self.save_h5_sample = False
        else:
            self.save_h5_sample = True
            self.dataset.close()
            self.dataset = h5py.File(self.dataset_name, "a")

    def _update_model_list(self):
        import time

        model_key_list = []
        while len(model_key_list) == 0:
            model_key_list = self._model_pool_api.pull_keys()
            if model_key_list is None:
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

    # randomly get a model from model pool with 80% probability for newest model and 20% probability for history models
    def _get_random_model(self):
        if self.agent_type in ["common_ai", "random"]:
            self.is_latest_model = False
            self._predictor._sess.run(self.model.init)
            self.model_version = ""
            return True

        self._update_model_list()
        rand_float = float(random.uniform(0, _G_RAND_MAX)) / float(_G_RAND_MAX)
        if rand_float <= _G_MODEL_UPDATE_RATIO:
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

    # handle the obs from gamecore, return action result
    @log_time("aiprocess_process")
    def process(self, state_dict, battle=False):
        feature_vec, legal_action = (
            state_dict["observation"],
            state_dict["legal_action"],
        )
        pred_ret = self._predict_process(feature_vec, legal_action)
        _, _, action, d_action = pred_ret
        if battle:
            return d_action
        return action, d_action, self._sample_process(state_dict, pred_ret)

    def _update_legal_action(self, original_la, actions):
        target_size = ModelConfig.LABEL_SIZE_LIST[-1]
        top_size = ModelConfig.LABEL_SIZE_LIST[0]
        original_la = np.array(original_la)
        fix_part = original_la[: -target_size * top_size]
        target_la = original_la[-target_size * top_size :]
        target_la = target_la.reshape([top_size, target_size])[actions[0]]
        return np.concatenate([fix_part, target_la], axis=0)

    # build samples from state infos
    def _sample_process(self, state_dict, pred_ret):
        # get is_train
        is_train = False
        req_pb = state_dict["req_pb"]
        for hero in req_pb.hero_list:
            if hero.camp == self.hero_camp:
                is_train = True if hero.hp > 0 else False

        frame_no = req_pb.frame_no
        feature_vec, reward, sub_action_mask = (
            state_dict["observation"],
            state_dict["reward"],
            state_dict["sub_action_mask"],
        )
        done = False
        prob, value, action, _ = pred_ret

        legal_action = self._update_legal_action(state_dict["legal_action"], action)
        keys = (
            "frame_no",
            "vec_feature",
            "legal_action",
            "action",
            "reward",
            "value",
            "prob",
            "sub_action",
            "lstm_cell",
            "lstm_hidden",
            "done",
            "is_train",
        )
        values = (
            frame_no,
            feature_vec,
            legal_action,
            action,
            reward[-1],
            value,
            prob,
            sub_action_mask,
            self.lstm_cell,
            self.lstm_hidden,
            done,
            is_train,
        )
        sample = dict(zip(keys, values))
        self.last_sample = sample

        if self.save_h5_sample:
            self._sample_process_for_saver(sample)
        return sample

    def _get_h5file_keys(self, h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    def _sample_process_for_saver(self, sample_dict):
        keys = ("frame_no", "vec_feature", "legal_action", "action", "reward", "done")
        keys_in_h5 = self._get_h5file_keys(self.dataset)
        if len(keys_in_h5) == 0:
            self.dataset.create_dataset(
                "frame_no",
                data=[[sample_dict["frame_no"]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )
            self.dataset.create_dataset(
                "observation",
                data=[sample_dict["vec_feature"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["vec_feature"])),
                chunks=True,
            )
            self.dataset.create_dataset(
                "legal_action",
                data=[sample_dict["legal_action"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["legal_action"])),
                chunks=True,
            )
            self.dataset.create_dataset(
                "action",
                data=[sample_dict["action"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["action"])),
                chunks=True,
            )
            self.dataset.create_dataset(
                "reward",
                data=[[sample_dict["reward"]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )
            self.dataset.create_dataset(
                "done",
                data=[[sample_dict["done"]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )

        else:
            for key, value in sample_dict.items():
                if key in keys:
                    key_dataset = key
                    if key_dataset == "vec_feature":
                        key_dataset = "observation"
                    self.dataset[key_dataset].resize(
                        (self.dataset[key_dataset].shape[0] + 1), axis=0
                    )
                    if isinstance(value, list):
                        self.dataset[key_dataset][-1] = value
                    else:
                        self.dataset[key_dataset][-1] = [value]

    # given the feature vec and legal_action,return output of the network
    def _predict_process(self, feature, legal_action):
        # put data to input
        input_list = cvt_tensor_to_infer_input(self.model.get_input_tensors())
        input_list[0].set_data(np.array(feature))
        input_list[1].set_data(np.array(legal_action))
        # input_list[2].set_data(label_list)
        input_list[2].set_data(self.lstm_cell)
        input_list[3].set_data(self.lstm_hidden)

        output_list = cvt_tensor_to_infer_output(self.model.get_output_tensors())
        output_list = self._predictor.inference(
            input_list=input_list, output_list=output_list
        )
        # cvt output dataxz
        np_output = cvt_infer_list_to_numpy_list(output_list)

        logits, value, self.lstm_cell, self.lstm_hidden = np_output[:4]

        prob, action, d_action = self._sample_masked_action(logits, legal_action)

        return prob, value, action, d_action  # prob: [[ ]], others: all 1D

    # get final executable actions
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [
            sum(self.label_size_list[: index + 1])
            for index in range(len(self.label_size_list))
        ]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits[0], label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],  # [12, 8]
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]  # [12]
        one_hot_actions = np.reshape(
            one_hot_actions, [self.label_size_list[0], 1]
        )  # [12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action  # [12]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        # target_legal_action = tf.gather(target_legal_action, action_idx, axis=1)
        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        # legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        # prob_list.append(probs)
        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        # tmp = tf.exp(tmp - tmp_max)* legal_action + _lsm_const_e
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        """
        Sample with probability, input probs should be 1D array
        """
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))

    def close(self):
        if self.dataset is not None:
            self.save_h5_sample = True
            self.dataset.close()
