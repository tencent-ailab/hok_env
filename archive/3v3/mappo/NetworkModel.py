import torch  # in place of tensorflow
import torch.nn as nn  # for builtin modules including Linear, Conv2d, MultiheadAttention, LayerNorm, etc
from torch.nn import ModuleDict  # for layer naming when nn.Sequential is not viable
import numpy as np  # for some basic dimention computation, might be redundent

from math import ceil, floor
from collections import OrderedDict

# typing
from torch import Tensor, LongTensor
from typing import Dict, List, Tuple
from ctypes import Union

from config.Config import Config
from config.DimConfig import DimConfig


##################
## Actual model ##
##################
class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()
        # feature configure parameter
        self.model_name = Config.NETWORK_NAME
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.target_embedding_dim = Config.TARGET_EMBEDDING_DIM
        self.hero_data_split_shape = Config.HERO_DATA_SPLIT_SHAPE
        self.hero_seri_vec_split_shape = Config.HERO_SERI_VEC_SPLIT_SHAPE
        self.hero_feature_img_channel = Config.HERO_FEATURE_IMG_CHANNEL
        self.hero_label_size_list = Config.HERO_LABEL_SIZE_LIST
        self.hero_is_reinforce_task_list = Config.HERO_IS_REINFORCE_TASK_LIST

        self.learning_rate = Config.INIT_LEARNING_RATE_START
        self.var_beta = Config.BETA_START

        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.min_policy = Config.MIN_POLICY
        self.embedding_trainable = False
        self.value_head_num = Config.VALUE_HEAD_NUM

        self.hero_num = 3
        self.hero_data_len = sum(Config.data_shapes[0])
        self.single_hero_feature_dim = int(DimConfig.DIM_OF_HERO_EMY[0])
        self.single_soldier_feature_dim = int(DimConfig.DIM_OF_SOLDIER_1_10[0])
        self.single_organ_feature_dim = int(DimConfig.DIM_OF_ORGAN_1_3[0])
        self.singel_monster_feature_dim = int(DimConfig.DIM_OF_MONSTER_1_20[0])

        self.global_feature_dim = int(np.sum(DimConfig.DIM_OF_GLOBAL_INFO))
        self.hero_main_feature_dim = int(DimConfig.DIM_OF_HERO_MAIN[0])

        self.all_hero_feature_dim = (
            int(np.sum(DimConfig.DIM_OF_HERO_FRD))
            + int(np.sum(DimConfig.DIM_OF_HERO_EMY))
            + int(np.sum(DimConfig.DIM_OF_HERO_MAIN))
        )
        self.all_soldier_feature_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)) + int(
            np.sum(DimConfig.DIM_OF_SOLDIER_11_20)
        )
        self.all_organ_feature_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1_3)) + int(
            np.sum(DimConfig.DIM_OF_ORGAN_4_6)
        )
        self.all_monster_feature_dim = int(np.sum(DimConfig.DIM_OF_MONSTER_1_20))

        # build network
        kernel_size_list = [(5, 5), (3, 3)]
        padding_list = ["same", "same"]
        channel_list = [self.hero_feature_img_channel[0][0], 18, 12]
        assert (
            len(channel_list) == len(kernel_size_list) + 1
        ), "channel list and kernel size list length mismatch"
        assert len(kernel_size_list) == len(
            padding_list
        ), "kernel size list and padding list length mismatch"

        """img_conv module"""
        self.conv_layers = nn.Sequential()
        for i, kernel_size in enumerate(kernel_size_list):
            is_last_layer = i == len(kernel_size_list) - 1
            conv_layer, _ = make_conv_layer(
                kernel_size, channel_list[i], channel_list[i + 1], padding_list[i]
            )
            self.conv_layers.add_module("img_feat_conv{0}".format(i + 1), conv_layer)

            if not is_last_layer:
                self.conv_layers.add_module("img_feat_relu{0}".format(i + 1), nn.ReLU())
                self.conv_layers.add_module(
                    "img_feat_maxpool{0}".format(i + 1), nn.MaxPool2d(3, 2)
                )

        """ hero_main module"""
        fc_hero_main_dim_list = [self.hero_main_feature_dim, 64, 64, 64, 64, 32]
        self.hero_main_mlp = MLP(fc_hero_main_dim_list, "hero_main_mlp")

        """ hero_share module"""
        fc_hero_dim_list = [self.single_hero_feature_dim, 512, 512, 128, 128, 64]
        self.hero_mlp = MLP(fc_hero_dim_list[:-1], "hero_mlp", non_linearity_last=True)
        self.hero_frd_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "hero_frd_fc",
                        make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1]),
                    )
                ]
            )
        )
        self.hero_emy_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "hero_emy_fc",
                        make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1]),
                    )
                ]
            )
        )

        """ soldier_share module"""
        ## first and second fc layers are shared by 2 soldier vecs
        fc_soldier_dim_list = [self.single_soldier_feature_dim, 64, 64, 64, 32]
        self.soldier_mlp = MLP(
            fc_soldier_dim_list[:-1], "soldier_mlp", non_linearity_last=True
        )
        ## the nn.Sequential is only for naming
        self.soldier_frd_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "soldier_frd_fc",
                        make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1]),
                    )
                ]
            )
        )
        self.soldier_emy_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "soldier_emy_fc",
                        make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1]),
                    )
                ]
            )
        )

        """ organ_share module"""
        fc_organ_dim_list = [self.single_organ_feature_dim, 64, 64, 64, 32]
        self.organ_mlp = MLP(
            fc_organ_dim_list[:-1], "organ_mlp", non_linearity_last=True
        )
        self.organ_frd_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "organ_frd_fc",
                        make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1]),
                    )
                ]
            )
        )
        self.organ_emy_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "organ_emy_fc",
                        make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1]),
                    )
                ]
            )
        )

        """ monster_share module"""
        fc_monster_dim_list = [self.singel_monster_feature_dim, 64, 64, 64, 32]
        self.monster_mlp = MLP(fc_monster_dim_list, "monster_mlp")

        """public concat"""
        fc_concat_dim_list = [1156, 256]
        self.concat_mlp = MLP(fc_concat_dim_list, "concat_mlp", non_linearity_last=True)

        self.label_mlp = ModuleDict(
            {
                "hero_label{0}_mlp".format(label_index): MLP(
                    [256, 64, self.hero_label_size_list[0][label_index]],
                    "hero_label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.hero_label_size_list[0]))
            }
        )

        self.value_mlp = MLP([256, 64, 1], "hero_value_mlp")
        # 全局critic
        self.value_mlp_all = MLP([640, 64, 1], "hero_value_mlp_all")

    def forward(self, data_list):
        all_hero_result_list = []
        hero_public_first_result_list = []
        hero_public_second_result_list = []
        for hero_index, hero_data in enumerate(data_list):
            hero_feature = hero_data[0]

            img_fet_dim = np.prod(self.hero_seri_vec_split_shape[hero_index][0])
            vec_fet_dim = np.prod(self.hero_seri_vec_split_shape[hero_index][1])
            feature_img, feature_vec = hero_feature.split(
                [img_fet_dim, vec_fet_dim], dim=1
            )

            feature_img_shape = list(self.hero_seri_vec_split_shape[0][0])
            feature_img_shape.insert(0, -1)  # (bs, c, h, w)
            feature_vec_shape = list(self.hero_seri_vec_split_shape[0][1])
            feature_vec_shape.insert(0, -1)

            _feature_img = feature_img.reshape(feature_img_shape)
            _feature_vec = feature_vec.reshape(feature_vec_shape)

            conv_result = self.conv_layers(_feature_img)
            flatten_conv_result = conv_result.flatten(start_dim=1)

            split_feature_vec = _feature_vec.split(
                [
                    self.all_hero_feature_dim,
                    self.all_soldier_feature_dim,
                    self.all_organ_feature_dim,
                    self.all_monster_feature_dim,
                    self.global_feature_dim,
                ],
                dim=1,
            )

            hero_vec_list = split_feature_vec[0].split(
                [
                    int(np.sum(DimConfig.DIM_OF_HERO_FRD)),
                    int(np.sum(DimConfig.DIM_OF_HERO_EMY)),
                    int(np.sum(DimConfig.DIM_OF_HERO_MAIN)),
                ],
                dim=1,
            )
            hero_frd = hero_vec_list[0].split(DimConfig.DIM_OF_HERO_FRD, dim=1)
            _hero_frd = torch.stack(hero_frd, dim=1)
            hero_emy = hero_vec_list[1].split(DimConfig.DIM_OF_HERO_EMY, dim=1)
            _hero_emy = torch.stack(hero_emy, dim=1)
            hero_main = hero_vec_list[2].split(DimConfig.DIM_OF_HERO_MAIN, dim=1)[0]

            soldier_vec_list = split_feature_vec[1].split(
                [
                    int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)),
                    int(np.sum(DimConfig.DIM_OF_SOLDIER_11_20)),
                ],
                dim=1,
            )
            soldier_1_10 = soldier_vec_list[0].split(
                DimConfig.DIM_OF_SOLDIER_1_10, dim=1
            )
            _soldier_1_10 = torch.stack(soldier_1_10, dim=1)
            soldier_11_20 = soldier_vec_list[0].split(
                DimConfig.DIM_OF_SOLDIER_11_20, dim=1
            )
            _soldier_11_20 = torch.stack(soldier_11_20, dim=1)

            organ_vec_list = split_feature_vec[2].split(
                [
                    int(np.sum(DimConfig.DIM_OF_ORGAN_1_3)),
                    int(np.sum(DimConfig.DIM_OF_ORGAN_4_6)),
                ],
                dim=1,
            )
            organ_1_3 = organ_vec_list[0].split(DimConfig.DIM_OF_ORGAN_1_3, dim=1)
            _organ_1_3 = torch.stack(organ_1_3, dim=1)
            organ_4_6 = organ_vec_list[1].split(DimConfig.DIM_OF_ORGAN_4_6, dim=1)
            _organ_4_6 = torch.stack(organ_4_6, dim=1)

            monster_vec_list = split_feature_vec[3]
            monster_1_20 = monster_vec_list.split(DimConfig.DIM_OF_MONSTER_1_20, dim=1)
            _monster_1_20 = torch.stack(monster_1_20, dim=1)

            global_info = split_feature_vec[4]

            """ real computations
            """
            # hero_share
            hero_frd_mlp_out = self.hero_mlp(_hero_frd)
            hero_frd_fc_out = self.hero_frd_fc(hero_frd_mlp_out)
            pool_frd_hero, _ = hero_frd_fc_out.max(dim=1)

            hero_emy_mlp_out = self.hero_mlp(_hero_emy)
            hero_emy_fc_out = self.hero_frd_fc(hero_emy_mlp_out)
            pool_emy_hero, _ = hero_emy_fc_out.max(dim=1)

            # soldier_share
            soldier_frd_mlp_out = self.soldier_mlp(_soldier_1_10)
            soldier_frd_fc_out = self.soldier_frd_fc(soldier_frd_mlp_out)
            # torch.max returns both the max values and the argmax indices
            pool_frd_soldier, _ = soldier_frd_fc_out.max(dim=1)

            soldier_emy_mlp_out = self.soldier_mlp(_soldier_11_20)
            soldier_emy_fc_out = self.soldier_emy_fc(soldier_emy_mlp_out)
            pool_emy_soldier, _ = soldier_emy_fc_out.max(dim=1)

            # monster_share
            monster_mlp_out = self.monster_mlp(_monster_1_20)
            pool_monster, _ = monster_mlp_out.max(dim=1)

            # organ_share
            organ_frd_mlp_out = self.organ_mlp(_organ_1_3)
            organ_frd_fc_out = self.organ_frd_fc(organ_frd_mlp_out)
            pool_frd_organ, _ = organ_frd_fc_out.max(dim=1)

            organ_emy_mlp_out = self.organ_mlp(_organ_4_6)
            organ_emy_fc_out = self.organ_emy_fc(organ_emy_mlp_out)
            pool_emy_organ, _ = organ_emy_fc_out.max(dim=1)

            # hero_main
            main_hero = self.hero_main_mlp(hero_main)

            concat_result = torch.cat(
                [
                    flatten_conv_result,
                    pool_frd_soldier,
                    pool_emy_soldier,
                    pool_monster,
                    pool_frd_organ,
                    pool_emy_organ,
                    main_hero,
                    pool_frd_hero,
                    pool_emy_hero,
                    global_info,
                ],
                dim=1,
            )

            fc_concat_result = self.concat_mlp(concat_result)
            hero_public_result = fc_concat_result.split([64, 192], dim=1)
            # shared encoding
            hero_public_first_result_list.append(hero_public_result[0])
            # individual encoding
            hero_public_second_result_list.append(hero_public_result[1])

        pool_hero_public, _ = torch.stack(hero_public_first_result_list, dim=1).max(
            dim=1
        )

        # 把pool_hero_public和三个英雄的hero_public_second_result_list直接连在一起

        fc_public_result_all = torch.cat([pool_hero_public], dim=1)
        for hero_index in range(self.hero_num):
            fc_public_result_all = torch.cat(
                [fc_public_result_all, hero_public_second_result_list[hero_index]],
                dim=1,
            )

        # fc_public_result_all = torch.cat([pool_hero_public, hero_public_second_result_list_all], dim=1)

        for hero_index in range(self.hero_num):
            hero_result_list = []
            fc_public_result = torch.cat(
                [pool_hero_public, hero_public_second_result_list[hero_index]], dim=1
            )
            for label_index, label_dim in enumerate(
                self.hero_label_size_list[hero_index]
            ):
                label_mlp_out = self.label_mlp["hero_label{0}_mlp".format(label_index)](
                    fc_public_result
                )
                hero_result_list.append(label_mlp_out)
            # hero_result_list.append(self.value_mlp(fc_public_result))
            hero_result_list.append(self.value_mlp_all(fc_public_result_all))

            all_hero_result_list.append(hero_result_list)

        return all_hero_result_list

    def compute_loss(self, data_list, rst_list):
        all_hero_loss_list = []
        total_loss = torch.tensor(0.0)
        # 所有英雄的hero_reward取出来求一下平均值

        tmp_hero_data = data_list[0][6:7]
        tmp_reward_list = [item.squeeze(1) for item in tmp_hero_data]
        tmp_reward = tmp_reward_list[0]
        # tmp_reward.zero_()

        for hero_index, hero_data in enumerate(data_list):
            if hero_index == 0:
                continue
            label_task_count = len(self.hero_label_size_list[hero_index])
            hero_reward_list = hero_data[
                (1 + label_task_count) : (2 + label_task_count)
            ]
            _hero_reward_list = [item.squeeze(1) for item in hero_reward_list]
            tmp_reward = tmp_reward + _hero_reward_list[0]

        reward_mean = torch.div(tmp_reward, 3)

        for hero_index, hero_data in enumerate(data_list):
            # _calculate_single_hero_loss
            label_task_count = len(self.hero_label_size_list[hero_index])
            hero_legal_action_flag_list = hero_data[1 : (1 + label_task_count)]
            hero_reward_list = hero_data[
                (1 + label_task_count) : (2 + label_task_count)
            ]
            hero_advantage = hero_data[(2 + label_task_count)]
            hero_action_list = hero_data[
                (3 + label_task_count) : (3 + label_task_count * 2)
            ]
            hero_probability_list = hero_data[
                (3 + label_task_count * 2) : (3 + label_task_count * 3)
            ]
            hero_frame_is_train = hero_data[(3 + label_task_count * 3)]
            hero_weight_list = hero_data[
                (4 + label_task_count * 3) : (4 + label_task_count * 4)
            ]
            hero_fc_label_result = rst_list[hero_index][:-1]
            hero_value_result = rst_list[hero_index][-1]
            hero_label_size_list = self.hero_label_size_list[hero_index]

            _hero_legal_action_flag_list = hero_legal_action_flag_list
            _hero_reward_list = [item.squeeze(1) for item in hero_reward_list]
            _hero_advantage = hero_advantage.squeeze(1)
            _hero_action_list = [item.squeeze(1) for item in hero_action_list]
            _hero_probability_list = hero_probability_list
            _hero_frame_is_train = hero_frame_is_train.squeeze(1)
            _hero_weight_list = [item.squeeze(1) for item in hero_weight_list]
            _hero_fc_label_result = hero_fc_label_result
            _hero_value_result = hero_value_result.squeeze(1)
            _hero_label_size_list = hero_label_size_list

            train_frame_count = _hero_frame_is_train.sum()
            train_frame_count = torch.maximum(
                train_frame_count, torch.tensor(1.0)
            )  # prevent division by 0

            # loss of value net
            # value_cost = torch.square((_hero_reward_list[0] - _hero_value_result)) * _hero_frame_is_train
            value_cost = (
                torch.square((reward_mean - _hero_value_result)) * _hero_frame_is_train
            )
            value_cost = 0.5 * value_cost.sum(0) / train_frame_count

            # for entropy loss calculate
            label_logits_subtract_max_list = []
            label_sum_exp_logits_list = []
            label_probability_list = []

            policy_cost = torch.tensor(0.0)
            for task_index, is_reinforce_task in enumerate(
                self.hero_is_reinforce_task_list[hero_index]
            ):
                if is_reinforce_task:
                    final_log_p = torch.tensor(0.0)
                    boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                    one_hot_actions = nn.functional.one_hot(
                        _hero_action_list[task_index].long(),
                        _hero_label_size_list[task_index],
                    )
                    legal_action_flag_list_max_mask = (
                        1 - _hero_legal_action_flag_list[task_index]
                    ) * boundary
                    label_logits_subtract_max = torch.clamp(
                        _hero_fc_label_result[task_index]
                        - torch.max(
                            _hero_fc_label_result[task_index]
                            - legal_action_flag_list_max_mask,
                            dim=1,
                            keepdim=True,
                        ).values,
                        -boundary,
                        1,
                    )

                    label_logits_subtract_max_list.append(label_logits_subtract_max)

                    label_exp_logits = (
                        _hero_legal_action_flag_list[task_index]
                        * torch.exp(label_logits_subtract_max)
                        + self.min_policy
                    )
                    label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                    label_sum_exp_logits_list.append(label_sum_exp_logits)

                    label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                    label_probability_list.append(label_probability)

                    policy_p = (one_hot_actions * label_probability).sum(1)
                    policy_log_p = torch.log(policy_p)
                    old_policy_p = (
                        one_hot_actions * _hero_probability_list[task_index]
                    ).sum(1)
                    old_policy_log_p = torch.log(old_policy_p)
                    final_log_p = final_log_p + policy_log_p - old_policy_log_p
                    ratio = torch.exp(final_log_p)
                    clip_ratio = ratio.clamp(0.0, 3.0)

                    surr1 = clip_ratio * _hero_advantage
                    surr2 = (
                        ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
                        * _hero_advantage
                    )
                    policy_cost = policy_cost - torch.sum(
                        torch.minimum(surr1, surr2)
                        * (_hero_weight_list[task_index].float())
                        * _hero_frame_is_train
                    ) / torch.maximum(
                        torch.sum(
                            (_hero_weight_list[task_index].float())
                            * _hero_frame_is_train
                        ),
                        torch.tensor(1.0),
                    )

            # cross entropy loss
            current_entropy_loss_index = 0
            entropy_loss_list = []
            for task_index, is_reinforce_task in enumerate(
                self.hero_is_reinforce_task_list[hero_index]
            ):
                if is_reinforce_task:
                    temp_entropy_loss = -torch.sum(
                        label_probability_list[current_entropy_loss_index]
                        * _hero_legal_action_flag_list[task_index]
                        * torch.log(label_probability_list[current_entropy_loss_index]),
                        dim=1,
                    )
                    temp_entropy_loss = -torch.sum(
                        (
                            temp_entropy_loss
                            * _hero_weight_list[task_index].float()
                            * _hero_frame_is_train
                        )
                    ) / torch.maximum(
                        torch.sum(
                            _hero_weight_list[task_index].float() * _hero_frame_is_train
                        ),
                        torch.tensor(1.0),
                    )  # add - because need to minize
                    entropy_loss_list.append(temp_entropy_loss)
                    current_entropy_loss_index = current_entropy_loss_index + 1
                else:
                    temp_entropy_loss = torch.tensor(0.0)
                    entropy_loss_list.append(temp_entropy_loss)
            entropy_cost = torch.tensor(0.0)
            for entropy_element in entropy_loss_list:
                entropy_cost = entropy_cost + entropy_element

            # cost_all
            cost_all = value_cost + policy_cost + self.var_beta * entropy_cost

            _hero_all_loss_list = [cost_all, value_cost, policy_cost, entropy_cost]

            total_loss = total_loss + _hero_all_loss_list[0]
            all_hero_loss_list.append(_hero_all_loss_list)
        return total_loss, [
            total_loss,
            [
                all_hero_loss_list[0][1],
                all_hero_loss_list[0][2],
                all_hero_loss_list[0][3],
            ],
            [
                all_hero_loss_list[1][1],
                all_hero_loss_list[1][2],
                all_hero_loss_list[1][3],
            ],
            [
                all_hero_loss_list[2][1],
                all_hero_loss_list[2][2],
                all_hero_loss_list[2][3],
            ],
        ]

    def format_data(self, datas):
        datas = datas.view(-1, self.hero_num, self.hero_data_len)
        data_list = datas.permute(1, 0, 2)

        hero_data_list = []
        for hero_index in range(self.hero_num):
            # calculate length of each frame
            hero_each_frame_data_length = np.sum(
                np.array(self.hero_data_split_shape[hero_index])
            )
            hero_sequence_data_length = (
                hero_each_frame_data_length * self.lstm_time_steps
            )
            hero_sequence_data_split_shape = [
                hero_sequence_data_length,
                self.lstm_unit_size,
                self.lstm_unit_size,
            ]

            sequence_data, lstm_cell_data, lstm_hidden_data = (
                data_list[hero_index]
                .float()
                .split(hero_sequence_data_split_shape, dim=1)
            )
            reshape_sequence_data = sequence_data.reshape(
                -1, hero_each_frame_data_length
            )
            hero_data = reshape_sequence_data.split(
                self.hero_data_split_shape[hero_index], dim=1
            )
            hero_data_list.append(hero_data)
        return hero_data_list


#######################0+
## Utility functions ##
#######################


def make_fc_layer(in_features: int, out_features: int):
    """Wrapper function to create and initialize a linear layer

    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``

    Returns:
        nn.Linear: the initialized linear layer
    """
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # nn.init.xavier_uniform_(fc_layer.weight)
    nn.init.orthogonal(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer


############################
## Building-block classes ##
############################
class MLP(nn.Module):
    """A simple multi-layer perceptron"""

    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        """Create a MLP object

        Args:
            fc_feat_dim_list (List[int]): ``in_features`` of the first linear layer followed by
                ``out_features`` of each linear layer
            name (str): human-friendly name, serving as prefix of each comprising layers
            non_linearity (nn.Module, optional): the activation function to use. Defaults to nn.ReLU.
            non_linearity_last (bool, optional): whether to append a activation function in the end.
                Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(
                    "{0}_non_linear{1}".format(name, i + 1), non_linearity()
                )

    def forward(self, data):
        return self.fc_layers(data)


def _compute_conv_out_shape(
    kernel_size: Tuple[int, int],
    padding: Tuple[int, int],
    input_shape: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
) -> Tuple[int, int]:
    """Compute the ouput shape of a convolution layer

    Args:
        kernel_size (Tuple[int, int]): kernel_size
        padding (Union[str, int]): either explicit padding size to add in both directions or
            padding scheme (either "same" or "valid)
        input_shape (Tuple[int, int]): [description]
        stride (Tuple[int, int], optional): [description]. Defaults to (1,1).

    Returns:
        Tuple[int, int]: height and width of the convolution ouput
    """
    out_x = (
        floor((input_shape[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0])
        + 1
    )
    out_y = (
        floor((input_shape[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1])
        + 1
    )
    return (out_x, out_y)


def make_conv_layer(
    kernel_size: Tuple[int, int],
    in_channels: int,
    out_channels: int,
    padding: str,
    stride: Tuple[int, int] = (1, 1),
    input_shape=None,
):
    """Wrapper function to create and initialize ``Conv2d`` layers. Returns output shape along the
        way if input shape is supplied. Add support for 'same' and 'valid' padding scheme (would
        be unnecessary if using pytorch 1.9.0 and higher).

    Args:
        kernel_size (Tuple[int, int]): height and width of the kernel
        in_channels (int): number of channels of the input image
        out_channels (int): number of channels of the convolution output
        padding (Union[str, Tuple[int, int]]): either explicit padding size to add in both
            directions or padding scheme (either "same" or "valid)
        stride (Union[int, Tuple[int, int]], optional): stride. Defaults to (1,1).
        input_shape (Tuple[int, int], optional): height and width of the input image. Defaults
            to None.

    Returns:
        (nn.Conv2d, Tuple[int, int]): the initialized convolution layer and the shape of the
            output image if input_shape is not None.
    """

    if isinstance(padding, str):
        assert padding in [
            "same",
            "valid",
        ], "Padding scheme must be either 'same' or 'valid'"
        if padding == "valid":
            padding = (0, 0)
        else:
            assert stride == 1 or (
                stride[0] == 1 and stride[1] == 1
            ), "Stride must be 1 when using 'same' as padding scheme"
            assert (
                kernel_size[0] % 2 and kernel_size[1] % 2
            ), "Currently, requiring kernel height and width to be odd for simplicity"
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # initialize weight and bias
    # nn.init.xavier_normal_(conv_layer.weight)
    nn.init.orthogonal(conv_layer.weight)
    nn.init.zeros_(conv_layer.bias)

    # compute output shape
    output_shape = None
    if input_shape:
        output_shape = _compute_conv_out_shape(
            kernel_size, padding, input_shape, stride
        )

    return conv_layer, output_shape
