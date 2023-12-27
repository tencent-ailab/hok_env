import tensorflow as tf
import numpy as np
from config.Config import Config
from config.DimConfig import DimConfig


class NetworkModel:
    def __init__(self):
        # feature configure parameter
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.hero_data_split_shape = Config.HERO_DATA_SPLIT_SHAPE
        self.hero_seri_vec_split_shape = Config.HERO_SERI_VEC_SPLIT_SHAPE
        self.hero_feature_img_channel = Config.HERO_FEATURE_IMG_CHANNEL
        self.hero_label_size_list = Config.HERO_LABEL_SIZE_LIST
        self.hero_is_reinforce_task_list = Config.HERO_IS_REINFORCE_TASK_LIST

        self.learning_rate = Config.INIT_LEARNING_RATE_START
        self.var_beta = Config.BETA_START

        self.clip_param = Config.CLIP_PARAM
        self.each_hero_loss_list = []
        self.restore_list = []
        self.min_policy = Config.MIN_POLICY
        self.embedding_trainable = True
        self.value_head_num = 1

        self.hero_num = 3
        self.hero_data_len = sum(Config.data_shapes[0])

    def build_graph(self, datas, global_step=None):
        datas = tf.reshape(datas, [-1, self.hero_num, self.hero_data_len])
        data_list = tf.transpose(datas, perm=[1, 0, 2])

        # new add 20180912
        each_hero_data_list = []
        for hero_index in range(self.hero_num):
            this_hero_data_list = self._split_data(
                tf.cast(data_list[hero_index], tf.float32),
                self.hero_data_split_shape,
            )
            each_hero_data_list.append(this_hero_data_list)
        # build network
        each_hero_fc_result_list = self._inference(each_hero_data_list)
        # calculate loss
        loss = self._calculate_loss(each_hero_data_list, each_hero_fc_result_list)
        # return
        return loss, [
            loss,
            [
                self.each_hero_loss_list[0][1],
                self.each_hero_loss_list[0][2],
                self.each_hero_loss_list[0][3],
            ],
            [
                self.each_hero_loss_list[1][1],
                self.each_hero_loss_list[1][2],
                self.each_hero_loss_list[1][3],
            ],
            [
                self.each_hero_loss_list[2][1],
                self.each_hero_loss_list[2][2],
                self.each_hero_loss_list[2][3],
            ],
        ]
        # return loss, [loss, [self.each_hero_loss_list[0][1], self.each_hero_loss_list[0][2], self.each_hero_loss_list[0][3], self.each_hero_loss_list[0][4], self.each_hero_loss_list[0][5],self.each_hero_loss_list[0][6],self.each_hero_loss_list[0][7]], [self.each_hero_loss_list[1][1], self.each_hero_loss_list[1][2], self.each_hero_loss_list[1][3],self.each_hero_loss_list[1][4], self.each_hero_loss_list[1][5], self.each_hero_loss_list[1][6], self.each_hero_loss_list[1][7]], [self.each_hero_loss_list[2][1], self.each_hero_loss_list[2][2], self.each_hero_loss_list[2][3],self.each_hero_loss_list[2][4], self.each_hero_loss_list[2][5], self.each_hero_loss_list[2][6], self.each_hero_loss_list[2][7]]],[]

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.00001)

    def _split_data(self, this_hero_data, this_hero_data_split_shape):
        # calculate length of each frame
        this_hero_each_frame_data_length = np.sum(np.array(this_hero_data_split_shape))
        this_hero_sequence_data_length = (
            this_hero_each_frame_data_length * self.lstm_time_steps
        )
        this_hero_sequence_data_split_shape = np.array(
            [this_hero_sequence_data_length, self.lstm_unit_size, self.lstm_unit_size]
        )
        sequence_data, lstm_cell_data, lstm_hidden_data = tf.split(
            this_hero_data, this_hero_sequence_data_split_shape, axis=1
        )
        reshape_sequence_data = tf.reshape(
            sequence_data, [-1, this_hero_each_frame_data_length]
        )
        sequence_this_hero_data_list = tf.split(
            reshape_sequence_data, np.array(this_hero_data_split_shape), axis=1
        )
        sequence_this_hero_data_list.append(lstm_cell_data)
        sequence_this_hero_data_list.append(lstm_hidden_data)
        return sequence_this_hero_data_list

    def _calculate_loss(self, each_hero_data_list, each_hero_fc_result_list):
        self.cost_all = tf.constant(0.0, dtype=tf.float32)
        for hero_index in range(len(each_hero_data_list)):
            this_hero_label_task_count = len(self.hero_label_size_list)
            this_hero_legal_action_flag_list = each_hero_data_list[hero_index][
                1 : (1 + this_hero_label_task_count)
            ]
            this_hero_reward_list = each_hero_data_list[hero_index][
                (1 + this_hero_label_task_count) : (
                    1 + this_hero_label_task_count + self.value_head_num
                )
            ]
            this_hero_advantage = each_hero_data_list[hero_index][
                (1 + this_hero_label_task_count + self.value_head_num)
            ]
            this_hero_action_list = each_hero_data_list[hero_index][
                (2 + this_hero_label_task_count + self.value_head_num) : (
                    2 + this_hero_label_task_count * 2 + self.value_head_num
                )
            ]
            this_hero_probability_list = each_hero_data_list[hero_index][
                (2 + this_hero_label_task_count * 2 + self.value_head_num) : (
                    2 + this_hero_label_task_count * 3 + self.value_head_num
                )
            ]
            this_hero_frame_is_train = each_hero_data_list[hero_index][
                (2 + this_hero_label_task_count * 3 + self.value_head_num)
            ]
            this_hero_weight_list = each_hero_data_list[hero_index][
                (3 + this_hero_label_task_count * 3 + self.value_head_num) : (
                    3 + this_hero_label_task_count * 4 + self.value_head_num
                )
            ]
            this_hero_fc_label_list = each_hero_fc_result_list[hero_index][
                : -self.value_head_num
            ]
            this_hero_value = each_hero_fc_result_list[hero_index][
                -self.value_head_num :
            ]
            this_hero_all_loss_list = self._calculate_single_hero_loss(
                this_hero_legal_action_flag_list,
                this_hero_reward_list,
                this_hero_advantage,
                this_hero_action_list,
                this_hero_probability_list,
                this_hero_frame_is_train,
                this_hero_fc_label_list,
                this_hero_value,
                self.hero_label_size_list,
                self.hero_is_reinforce_task_list[hero_index],
                this_hero_weight_list,
            )
            self.cost_all = self.cost_all + this_hero_all_loss_list[0]
            self.each_hero_loss_list.append(this_hero_all_loss_list)
        return self.cost_all

    def _squeeze_tensor(
        self,
        unsqueeze_reward_list,
        unsqueeze_advantage,
        unsqueeze_label_list,
        unsqueeze_frame_is_train,
        unsqueeze_weight_list,
    ):
        reward_list = []
        for ele in unsqueeze_reward_list:
            reward_list.append(tf.squeeze(ele, axis=[1]))
        advantage = tf.squeeze(unsqueeze_advantage, axis=[1])
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(tf.squeeze(ele, axis=[1]))
        frame_is_train = tf.squeeze(unsqueeze_frame_is_train, axis=[1])
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(tf.squeeze(weight, axis=[1]))
        return reward_list, advantage, label_list, frame_is_train, weight_list

    def _calculate_single_hero_loss(
        self,
        legal_action_flag_list,
        unsqueeze_reward_list,
        unsqueeze_advantage,
        unsqueeze_label_list,
        old_label_probability_list,
        unsqueeze_frame_is_train,
        fc2_label_list,
        fc2_value_result,
        label_size_list,
        is_reinforce_task_list,
        unsqueeze_weight_list,
    ):

        (
            reward_list,
            advantage,
            label_list,
            frame_is_train,
            weight_list,
        ) = self._squeeze_tensor(
            unsqueeze_reward_list,
            unsqueeze_advantage,
            unsqueeze_label_list,
            unsqueeze_frame_is_train,
            unsqueeze_weight_list,
        )

        train_frame_count = tf.reduce_sum(frame_is_train)
        train_frame_count = tf.maximum(train_frame_count, 1.0)  # prevent division by 0

        # loss of value net
        value_cost = []
        for value_index in range(self.value_head_num):
            fc2_value_result_squeezed = tf.squeeze(
                fc2_value_result[value_index], axis=[1]
            )
            value_cost_tmp = 0.5 * tf.reduce_mean(
                tf.square(reward_list[value_index] - fc2_value_result_squeezed), axis=0
            )
            value_cost.append(value_cost_tmp)
        # value_cost = tf.square((reward - fc2_value_result_squeezed)) * frame_is_train
        # value_cost = 0.5 * tf.reduce_sum(value_cost, axis=0) / train_frame_count

        # for entropy loss calculate
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []

        # policy loss: ppo clip loss
        policy_cost = tf.constant(0.0, dtype=tf.float32)
        for task_index in range(len(is_reinforce_task_list)):
            if is_reinforce_task_list[task_index]:
                final_log_p = tf.constant(0.0, dtype=tf.float32)
                one_hot_actions = tf.one_hot(
                    tf.to_int32(label_list[task_index]), label_size_list[task_index]
                )
                legal_action_flag_list_max_mask = (
                    1 - legal_action_flag_list[task_index]
                ) * tf.pow(10.0, 20.0)
                label_logits_subtract_max = tf.clip_by_value(
                    (
                        fc2_label_list[task_index]
                        - tf.reduce_max(
                            fc2_label_list[task_index]
                            - legal_action_flag_list_max_mask,
                            axis=1,
                            keep_dims=True,
                        )
                    ),
                    -tf.pow(10.0, 20.0),
                    1,
                )
                label_logits_subtract_max_list.append(label_logits_subtract_max)
                label_exp_logits = (
                    legal_action_flag_list[task_index]
                    * tf.exp(label_logits_subtract_max)
                    + self.min_policy
                )
                label_sum_exp_logits = tf.reduce_sum(
                    label_exp_logits, axis=1, keep_dims=True
                )
                label_sum_exp_logits_list.append(label_sum_exp_logits)
                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)
                policy_p = tf.reduce_sum(one_hot_actions * label_probability, axis=1)
                policy_log_p = tf.log(policy_p)
                # policy_log_p = tf.to_float(weight_list[task_index]) * policy_log_p
                old_policy_p = tf.reduce_sum(
                    one_hot_actions * old_label_probability_list[task_index], axis=1
                )
                old_policy_log_p = tf.log(old_policy_p)
                # old_policy_log_p = tf.to_float(weight_list[task_index]) * old_policy_log_p
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = tf.exp(final_log_p)
                clip_ratio = tf.clip_by_value(ratio, 0.0, 3.0)
                surr1 = clip_ratio * advantage
                surr2 = (
                    tf.clip_by_value(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * advantage
                )
                policy_cost = policy_cost - tf.reduce_sum(
                    tf.minimum(surr1, surr2)
                    * tf.to_float(weight_list[task_index])
                    * frame_is_train
                ) / tf.maximum(
                    tf.reduce_sum(
                        tf.to_float(weight_list[task_index]) * frame_is_train
                    ),
                    1.0,
                )
                # policy_cost = - tf.reduce_sum(policy_cost, axis=0) / train_frame_count # CLIP loss, add - because need to minize

        # cross entropy loss
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(is_reinforce_task_list)):
            if is_reinforce_task_list[task_index]:
                # temp_entropy_loss = -tf.reduce_sum(label_probability_list[current_entropy_loss_index] * (label_logits_subtract_max_list[current_entropy_loss_index] - tf.log(label_sum_exp_logits_list[current_entropy_loss_index])), axis=1)
                temp_entropy_loss = -tf.reduce_sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * tf.log(label_probability_list[current_entropy_loss_index]),
                    axis=1,
                )
                temp_entropy_loss = -tf.reduce_sum(
                    (
                        temp_entropy_loss
                        * tf.to_float(weight_list[task_index])
                        * frame_is_train
                    )
                ) / tf.maximum(
                    tf.reduce_sum(
                        tf.to_float(weight_list[task_index]) * frame_is_train
                    ),
                    1.0,
                )  # add - because need to minize
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = tf.constant(0.0, dtype=tf.float32)
                entropy_loss_list.append(temp_entropy_loss)
        entropy_cost = tf.constant(0.0, dtype=tf.float32)
        for entropy_element in entropy_loss_list:
            entropy_cost = entropy_cost + entropy_element

        value_cost_all = tf.constant(0.0, dtype=tf.float32)
        for value_ele in value_cost:
            value_cost_all += value_ele
        # cost_all
        cost_all = value_cost_all + policy_cost + self.var_beta * entropy_cost

        return cost_all, value_cost[0], policy_cost, entropy_cost

    def _inference(self, each_hero_data_list):
        # share weight
        with tf.variable_scope("hero_all_start"):
            conv1_kernel = self._conv_weight_variable(
                shape=[5, 5, self.hero_feature_img_channel, 18],
                name="hero_conv1_kernel",
            )
            conv1_bias = self._bias_variable(shape=[18], name="hero_conv1_bias")

            conv2_kernel = self._conv_weight_variable(
                shape=[3, 3, 18, 12], name="hero_conv2_kernel"
            )
            conv2_bias = self._bias_variable(shape=[12], name="hero_conv2_bias")

            with tf.variable_scope("share"):
                #### soldier
                fc1_soldier_weight = self._fc_weight_variable(
                    shape=[25, 64], name="fc1_soldier_weight"
                )
                fc1_soldier_bias = self._bias_variable(
                    shape=[64], name="fc1_soldier_bias"
                )
                fc2_soldier_weight = self._fc_weight_variable(
                    shape=[64, 64], name="fc2_soldier_weight"
                )
                fc2_soldier_bias = self._bias_variable(
                    shape=[64], name="fc2_soldier_bias"
                )

                fc2_soldier_weight_add = self._fc_weight_variable(
                    shape=[64, 64], name="fc2_soldier_weight_add"
                )
                fc2_soldier_bias_add = self._bias_variable(
                    shape=[64], name="fc2_soldier_bias_add"
                )

                fc3_soldier_weight_frd = self._fc_weight_variable(
                    shape=[64, 32], name="fc3_soldier_2_frd_weight"
                )
                fc3_soldier_bias_frd = self._bias_variable(
                    shape=[32], name="fc3_soldier_2_frd_bias"
                )
                fc3_soldier_weight_enemy = self._fc_weight_variable(
                    shape=[64, 32], name="fc3_soldier_2_enemy_weight"
                )
                fc3_soldier_bias_enemy = self._bias_variable(
                    shape=[32], name="fc3_soldier_2_enemy_bias"
                )

                ### monster
                fc1_monster_weight = self._fc_weight_variable(
                    shape=[28, 64], name="fc1_monster_weight"
                )
                fc1_monster_bias = self._bias_variable(
                    shape=[64], name="fc1_monster_bias"
                )

                fc1_monster_weight_add = self._fc_weight_variable(
                    shape=[64, 64], name="fc1_monster_weight_add"
                )
                fc1_monster_bias_add = self._bias_variable(
                    shape=[64], name="fc1_monster_bias_add"
                )

                fc2_monster_weight = self._fc_weight_variable(
                    shape=[64, 32], name="fc2_monster_weight"
                )
                fc2_monster_bias = self._bias_variable(
                    shape=[32], name="fc2_monster_bias"
                )

                fc3_monster_weight = self._fc_weight_variable(
                    shape=[32, 32], name="fc3_monster_weight"
                )
                fc3_monster_bias = self._bias_variable(
                    shape=[32], name="fc3_monster_bias"
                )

                #### organ
                fc1_organ_weight = self._fc_weight_variable(
                    shape=[29, 64], name="fc1_organ_weight"
                )
                fc1_organ_bias = self._bias_variable(shape=[64], name="fc1_organ_bias")
                fc2_organ_weight = self._fc_weight_variable(
                    shape=[64, 64], name="fc2_organ_weight"
                )
                fc2_organ_bias = self._bias_variable(shape=[64], name="fc2_organ_bias")

                fc2_organ_weight_add = self._fc_weight_variable(
                    shape=[64, 64], name="fc2_organ_weight_add"
                )
                fc2_organ_bias_add = self._bias_variable(
                    shape=[64], name="fc2_organ_bias_add"
                )

                fc3_organ_weight_frd = self._fc_weight_variable(
                    shape=[64, 32], name="fc3_organ_1_weight"
                )
                fc3_organ_bias_frd = self._bias_variable(
                    shape=[32], name="fc3_organ_1_bias"
                )
                fc3_organ_weight_enemy = self._fc_weight_variable(
                    shape=[64, 32], name="fc3_organ_2_weight"
                )
                fc3_organ_bias_enemy = self._bias_variable(
                    shape=[32], name="fc3_organ_2_bias"
                )

            #### main_hero
            fc1_hero_weight = self._fc_weight_variable(
                shape=[44, 64], name="fc1_hero_main_weight"
            )
            fc1_hero_bias = self._bias_variable(shape=[64], name="fc1_hero_main_bias")

            fc1_hero_weight_add = self._fc_weight_variable(
                shape=[64, 64], name="fc1_hero_main_weight_add"
            )
            fc1_hero_bias_add = self._bias_variable(
                shape=[64], name="fc1_hero_main_bias_add"
            )

            fc2_hero_weight = self._fc_weight_variable(
                shape=[64, 64], name="fc2_hero_main_weight"
            )
            fc2_hero_bias = self._bias_variable(shape=[64], name="fc2_hero_main_bias")

            fc2_hero_weight_add = self._fc_weight_variable(
                shape=[64, 64], name="fc2_hero_main_weight_add"
            )
            fc2_hero_bias_add = self._bias_variable(
                shape=[64], name="fc2_hero_main_bias_add"
            )

            fc3_hero_weight = self._fc_weight_variable(
                shape=[64, 32], name="fc3_hero_main_weight"
            )
            fc3_hero_bias = self._bias_variable(shape=[32], name="fc3_hero_main_bias")

            # 3v3 delete spec hero

            with tf.variable_scope("share"):
                #### ten_hero
                fc1_hero_weight_ten_hero = self._fc_weight_variable(
                    shape=[251, 512], name="fc1_hero_weight"
                )
                fc1_hero_bias_ten_hero = self._bias_variable(
                    shape=[512], name="fc1_hero_bias"
                )

                fc1_hero_weight_ten_hero_add = self._fc_weight_variable(
                    shape=[512, 512], name="fc1_hero_weight_add"
                )
                fc1_hero_bias_ten_hero_add = self._bias_variable(
                    shape=[512], name="fc1_hero_bias_add"
                )

                fc2_hero_weight_ten_hero = self._fc_weight_variable(
                    shape=[512, 128], name="fc2_hero_weight"
                )
                fc2_hero_bias_ten_hero = self._bias_variable(
                    shape=[128], name="fc2_hero_bias"
                )

                fc2_hero_weight_ten_hero_add = self._fc_weight_variable(
                    shape=[128, 128], name="fc2_hero_weight_add"
                )
                fc2_hero_bias_ten_hero_add = self._bias_variable(
                    shape=[128], name="fc2_hero_bias_add"
                )

                fc3_hero_weight_frd_five_hero = self._fc_weight_variable(
                    shape=[128, 64], name="fc3_hero_frd_weight"
                )
                fc3_hero_bias_frd_five_hero = self._bias_variable(
                    shape=[64], name="fc3_hero_frd_bias"
                )
                fc3_hero_weight_enemy_five_hero = self._fc_weight_variable(
                    shape=[128, 64], name="fc3_hero_emy_weight"
                )
                fc3_hero_bias_enemy_five_hero = self._bias_variable(
                    shape=[64], name="fc3_hero_emy_bias"
                )
                # 3v3 delete embedding

            # before max_pooling
            fc_public_weight = self._fc_weight_variable(
                shape=[1156, 256], name="hero_fc_public_weight"
            )
            fc_public_bias = self._bias_variable(
                shape=[256], name="hero_fc_public_bias"
            )

        # new add 20180912
        each_hero_fc_result_list = []
        hero_public_first_result_list = []
        hero_public_second_result_list = []
        # each_hero_embedding_list = []
        # hero_vision_result_list = []
        for hero_index in range(len(each_hero_data_list)):
            with tf.variable_scope("hero_%d" % hero_index, reuse=tf.AUTO_REUSE):
                # model design
                split_feature_img, split_feature_vec = tf.split(
                    each_hero_data_list[hero_index][0],
                    [
                        np.prod(self.hero_seri_vec_split_shape[0]),
                        np.prod(self.hero_seri_vec_split_shape[1]),
                    ],
                    axis=1,
                )
                feature_img_shape = list(self.hero_seri_vec_split_shape[0])
                feature_img_shape.insert(0, -1)
                feature_img = tf.reshape(split_feature_img, feature_img_shape)
                feature_vec_shape = list(self.hero_seri_vec_split_shape[1])
                feature_vec_shape.insert(0, -1)
                feature_vec = tf.reshape(split_feature_vec, feature_vec_shape)
                feature_img = tf.identity(
                    feature_img, name="hero%d_feature_img" % hero_index
                )
                feature_vec = tf.identity(
                    feature_vec, name="hero%d_feature_vec" % hero_index
                )

                with tf.variable_scope("hero%d_transpose" % hero_index):
                    transpose_feature_img = tf.transpose(
                        feature_img,
                        perm=[0, 2, 3, 1],
                        name="hero%d_transpose_feature_img" % hero_index,
                    )

                with tf.variable_scope("hero%d_conv1" % hero_index):
                    conv1_result = tf.nn.relu(
                        (
                            tf.nn.conv2d(
                                transpose_feature_img,
                                conv1_kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                            )
                            + conv1_bias
                        ),
                        name="hero%d_conv1_result" % hero_index,
                    )
                    pool_conv1_result = tf.nn.max_pool(
                        conv1_result,
                        [1, 3, 3, 1],
                        [1, 2, 2, 1],
                        padding="VALID",
                        name="hero%d_pool_conv1_result" % hero_index,
                    )

                with tf.variable_scope("hero%d_conv2" % hero_index):
                    temp_conv2_result = tf.nn.bias_add(
                        tf.nn.conv2d(
                            pool_conv1_result,
                            conv2_kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                        ),
                        conv2_bias,
                        name="hero%d_temp_conv2_result" % hero_index,
                    )
                    conv2_result = tf.transpose(
                        temp_conv2_result,
                        perm=[0, 3, 1, 2],
                        name="hero%d_conv2_result" % hero_index,
                    )

                with tf.variable_scope("hero%d_flatten_conv2" % hero_index):
                    conv2_dim = int(np.prod(conv2_result.get_shape()[1:]))
                    flatten_conv2_result = tf.reshape(
                        conv2_result,
                        shape=[-1, conv2_dim],
                        name="hero%d_flatten_conv2_result" % hero_index,
                    )

                hero_dim = (
                    int(np.sum(DimConfig.DIM_OF_HERO_FRD))
                    + int(np.sum(DimConfig.DIM_OF_HERO_EMY))
                    + int(np.sum(DimConfig.DIM_OF_HERO_MAIN))
                )
                soldier_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)) + int(
                    np.sum(DimConfig.DIM_OF_SOLDIER_11_20)
                )
                organ_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1_3)) + int(
                    np.sum(DimConfig.DIM_OF_ORGAN_4_6)
                )
                monster_dim = int(np.sum(DimConfig.DIM_OF_MONSTER_1_20))
                global_info_dim = int(np.sum(DimConfig.DIM_OF_GLOBAL_INFO))

                # 3V3 delete
                with tf.variable_scope("hero%d_feature_vec_split" % hero_index):
                    feature_vec_split_list = tf.split(
                        feature_vec,
                        [
                            hero_dim,
                            soldier_dim,
                            organ_dim,
                            monster_dim,
                            global_info_dim,
                        ],
                        axis=1,
                    )
                    hero_vec_list = tf.split(
                        feature_vec_split_list[0],
                        [
                            int(np.sum(DimConfig.DIM_OF_HERO_FRD)),
                            int(np.sum(DimConfig.DIM_OF_HERO_EMY)),
                            int(np.sum(DimConfig.DIM_OF_HERO_MAIN)),
                        ],
                        axis=1,
                    )
                    soldier_vec_list = tf.split(
                        feature_vec_split_list[1],
                        [
                            int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)),
                            int(np.sum(DimConfig.DIM_OF_SOLDIER_11_20)),
                        ],
                        axis=1,
                    )
                    organ_vec_list = tf.split(
                        feature_vec_split_list[2],
                        [
                            int(np.sum(DimConfig.DIM_OF_ORGAN_1_3)),
                            int(np.sum(DimConfig.DIM_OF_ORGAN_4_6)),
                        ],
                        axis=1,
                    )
                    monster_vec_list = feature_vec_split_list[3]

                    soldier_1_10 = tf.split(
                        soldier_vec_list[0], DimConfig.DIM_OF_SOLDIER_1_10, axis=1
                    )
                    soldier_11_20 = tf.split(
                        soldier_vec_list[1], DimConfig.DIM_OF_SOLDIER_11_20, axis=1
                    )
                    monster_1_20 = tf.split(
                        monster_vec_list, DimConfig.DIM_OF_MONSTER_1_20, axis=1
                    )
                    organ_1_3 = tf.split(
                        organ_vec_list[0], DimConfig.DIM_OF_ORGAN_1_3, axis=1
                    )
                    organ_4_6 = tf.split(
                        organ_vec_list[1], DimConfig.DIM_OF_ORGAN_4_6, axis=1
                    )
                    hero_frd = tf.split(
                        hero_vec_list[0], DimConfig.DIM_OF_HERO_FRD, axis=1
                    )
                    hero_emy = tf.split(
                        hero_vec_list[1], DimConfig.DIM_OF_HERO_EMY, axis=1
                    )
                    hero_main = tf.split(
                        hero_vec_list[2], DimConfig.DIM_OF_HERO_MAIN, axis=1
                    )
                    global_info = feature_vec_split_list[4]
                    # 3v3 delete spec hero

                    # 3v3 delete emb

                    # 3v3 after delete, index change

                # 3v3 delete vision

                with tf.variable_scope(
                    "hero%d_hero_share" % hero_index, reuse=tf.AUTO_REUSE
                ):
                    # 3v3 delete emb

                    hero_frd_result_list = []
                    for index in range(len(hero_frd)):
                        # 3v3 delete skill/equi emb
                        # hero state
                        vec_fc1_input_dim = int(
                            np.prod(hero_frd[index].get_shape()[1:])
                        )
                        fc1_hero_result = tf.nn.relu(
                            (
                                tf.matmul(hero_frd[index], fc1_hero_weight_ten_hero)
                                + fc1_hero_bias_ten_hero
                            ),
                            name="hero%d_fc1_hero_frd_result_%d" % (hero_index, index),
                        )
                        fc1_hero_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc1_hero_result, fc1_hero_weight_ten_hero_add)
                                + fc1_hero_bias_ten_hero_add
                            ),
                            name="hero%d_fc1_hero_frd_result_%d_add"
                            % (hero_index, index),
                        )
                        fc2_hero_result = tf.nn.relu(
                            (
                                tf.matmul(fc1_hero_result_add, fc2_hero_weight_ten_hero)
                                + fc2_hero_bias_ten_hero
                            ),
                            name="hero%d_fc2_hero_frd_result_%d" % (hero_index, index),
                        )
                        fc2_hero_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc2_hero_result, fc2_hero_weight_ten_hero_add)
                                + fc2_hero_bias_ten_hero_add
                            ),
                            name="hero%d_fc2_hero_frd_result_%d_add"
                            % (hero_index, index),
                        )
                        # concat all hero feat
                        # hero_frd_concat = tf.concat([reshape_pool_Skill, reshape_pool_Equip, fc2_hero_result_add], axis=1, name="hero_frd_concat")
                        fc3_hero_result = tf.add(
                            tf.matmul(
                                fc2_hero_result_add, fc3_hero_weight_frd_five_hero
                            ),
                            fc3_hero_bias_frd_five_hero,
                            name="hero%d_fc3_hero_frd_result_%d" % (hero_index, index),
                        )
                        # hero_embedding
                        split_0, split_1 = tf.split(fc3_hero_result, [16, 48], axis=1)

                        hero_frd_result_list.append(fc3_hero_result)
                    hero_frd_concat_result = tf.concat(
                        hero_frd_result_list,
                        axis=1,
                        name="hero%d_hero_frd_concat_result" % hero_index,
                    )

                    reshape_hero_frd = tf.reshape(
                        hero_frd_concat_result,
                        shape=[-1, 3, 64, 1],
                        name="hero%d_reshape_hero_frd" % hero_index,
                    )
                    pool_hero_frd = tf.nn.max_pool(
                        reshape_hero_frd,
                        [1, 3, 1, 1],
                        [1, 1, 1, 1],
                        padding="VALID",
                        name="hero%d_pool_hero_frd" % hero_index,
                    )
                    output_dim = int(np.prod(pool_hero_frd.get_shape()[1:]))
                    reshape_pool_hero_frd = tf.reshape(
                        pool_hero_frd,
                        shape=[-1, output_dim],
                        name="hero%d_reshape_pool_hero_frd" % hero_index,
                    )

                    hero_emy_result_list = []
                    for index in range(len(hero_emy)):
                        # 3v3 delete skill/ equip emb
                        # hero state
                        vec_fc1_input_dim = int(
                            np.prod(hero_emy[index].get_shape()[1:])
                        )
                        fc1_hero_result = tf.nn.relu(
                            (
                                tf.matmul(hero_emy[index], fc1_hero_weight_ten_hero)
                                + fc1_hero_bias_ten_hero
                            ),
                            name="hero%d_fc1_hero_emy_result_%d" % (hero_index, index),
                        )
                        fc1_hero_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc1_hero_result, fc1_hero_weight_ten_hero_add)
                                + fc1_hero_bias_ten_hero_add
                            ),
                            name="hero%d_fc1_hero_emy_result_%d_add"
                            % (hero_index, index),
                        )
                        fc2_hero_result = tf.nn.relu(
                            (
                                tf.matmul(fc1_hero_result_add, fc2_hero_weight_ten_hero)
                                + fc2_hero_bias_ten_hero
                            ),
                            name="hero%d_fc2_hero_emy_result_%d" % (hero_index, index),
                        )
                        fc2_hero_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc2_hero_result, fc2_hero_weight_ten_hero_add)
                                + fc2_hero_bias_ten_hero_add
                            ),
                            name="hero%d_fc2_hero_emy_result_%d_add"
                            % (hero_index, index),
                        )
                        # concat all hero feat
                        # hero_emy_concat = tf.concat([reshape_pool_Skill, reshape_pool_Equip, fc2_hero_result_add], axis=1, name="hero_emy_concat")
                        fc3_hero_result = tf.add(
                            tf.matmul(
                                fc2_hero_result_add, fc3_hero_weight_enemy_five_hero
                            ),
                            fc3_hero_bias_enemy_five_hero,
                            name="hero%d_fc3_hero_emy_result_%d" % (hero_index, index),
                        )

                        # hero_embedding
                        split_0, split_1 = tf.split(fc3_hero_result, [16, 48], axis=1)

                        hero_emy_result_list.append(fc3_hero_result)

                    hero_emy_concat_result = tf.concat(
                        hero_emy_result_list,
                        axis=1,
                        name="hero%d_hero_emy_concat_result" % hero_index,
                    )

                    reshape_hero_emy = tf.reshape(
                        hero_emy_concat_result,
                        shape=[-1, 3, 64, 1],
                        name="hero%d_reshape_hero_emy" % hero_index,
                    )
                    pool_hero_emy = tf.nn.max_pool(
                        reshape_hero_emy,
                        [1, 3, 1, 1],
                        [1, 1, 1, 1],
                        padding="VALID",
                        name="hero%d_pool_hero_emy" % hero_index,
                    )
                    output_dim = int(np.prod(pool_hero_emy.get_shape()[1:]))
                    reshape_pool_hero_emy = tf.reshape(
                        pool_hero_emy,
                        shape=[-1, output_dim],
                        name="hero%d_reshape_pool_hero_emy" % hero_index,
                    )

                with tf.variable_scope(
                    "hero%d_hero_main" % hero_index, reuse=tf.AUTO_REUSE
                ):
                    hero_main_result_list = []
                    for index in range(len(hero_main)):
                        vec_fc1_input_dim = int(
                            np.prod(hero_main[index].get_shape()[1:])
                        )
                        fc1_hero_result = tf.nn.relu(
                            (
                                tf.matmul(hero_main[index], fc1_hero_weight)
                                + fc1_hero_bias
                            ),
                            name="hero%d_fc1_hero_main_result_%d" % (hero_index, index),
                        )
                        fc1_hero_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc1_hero_result, fc1_hero_weight_add)
                                + fc1_hero_bias_add
                            ),
                            name="hero%d_fc1_hero_main_result_%d_add"
                            % (hero_index, index),
                        )
                        fc2_hero_result = tf.nn.relu(
                            (
                                tf.matmul(fc1_hero_result_add, fc2_hero_weight)
                                + fc2_hero_bias
                            ),
                            name="hero%d_fc2_hero_main_result_%d" % (hero_index, index),
                        )
                        fc2_hero_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc2_hero_result, fc2_hero_weight_add)
                                + fc2_hero_bias_add
                            ),
                            name="hero%d_fc2_hero_main_result_%d_add"
                            % (hero_index, index),
                        )
                        fc3_hero_result = tf.add(
                            tf.matmul(fc2_hero_result_add, fc3_hero_weight),
                            fc3_hero_bias,
                            name="hero%d_fc3_hero_main_result_%d" % (hero_index, index),
                        )

                    hero_main_concat_result = fc3_hero_result
                    # main_hero_embedding
                    split_0, split_1 = tf.split(
                        hero_main_concat_result, [16, 16], axis=1
                    )

                with tf.variable_scope(
                    "hero%d_monster_share" % hero_index, reuse=tf.AUTO_REUSE
                ):
                    monster_result_list = []
                    for index in range(len(monster_1_20)):
                        vec_fc1_input_dim = int(
                            np.prod(monster_1_20[index].get_shape()[1:])
                        )
                        fc1_monster_result = tf.nn.relu(
                            (
                                tf.matmul(monster_1_20[index], fc1_monster_weight)
                                + fc1_monster_bias
                            ),
                            name="hero%d_fc1_monster_result_%d" % (hero_index, index),
                        )
                        fc1_monster_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc1_monster_result, fc1_monster_weight_add)
                                + fc1_monster_bias_add
                            ),
                            name="hero%d_fc1_monster_result_%d_add"
                            % (hero_index, index),
                        )
                        fc2_monster_result = tf.nn.relu(
                            (
                                tf.matmul(fc1_monster_result_add, fc2_monster_weight)
                                + fc2_monster_bias
                            ),
                            name="hero%d_fc2_monster_result_%d" % (hero_index, index),
                        )
                        fc3_monster_result = tf.add(
                            tf.matmul(fc2_monster_result, fc3_monster_weight),
                            fc3_monster_bias,
                            name="hero%d_fc3_monster_result_%d" % (hero_index, index),
                        )

                        monster_result_list.append(fc3_monster_result)
                    monster_concat_result = tf.concat(
                        monster_result_list,
                        axis=1,
                        name="hero%d_monster_concat_result" % hero_index,
                    )

                    reshape_monster = tf.reshape(
                        monster_concat_result,
                        shape=[-1, 20, 32, 1],
                        name="hero%d_reshape_monster" % hero_index,
                    )
                    pool_monster = tf.nn.max_pool(
                        reshape_monster,
                        [1, 20, 1, 1],
                        [1, 1, 1, 1],
                        padding="VALID",
                        name="hero%d_pool_monster" % hero_index,
                    )
                    output_dim = int(np.prod(pool_monster.get_shape()[1:]))
                    reshape_pool_monster = tf.reshape(
                        pool_monster,
                        shape=[-1, output_dim],
                        name="hero%d_reshape_pool_monster" % hero_index,
                    )

                with tf.variable_scope(
                    "hero%d_soldier_share" % hero_index, reuse=tf.AUTO_REUSE
                ):
                    soldier_1_result_list = []
                    for index in range(len(soldier_1_10)):
                        vec_fc1_input_dim = int(
                            np.prod(soldier_1_10[index].get_shape()[1:])
                        )
                        fc1_soldier_result = tf.nn.relu(
                            (
                                tf.matmul(soldier_1_10[index], fc1_soldier_weight)
                                + fc1_soldier_bias
                            ),
                            name="hero%d_fc1_soldier_1_result_%d" % (hero_index, index),
                        )
                        fc2_soldier_result = tf.nn.relu(
                            (
                                tf.matmul(fc1_soldier_result, fc2_soldier_weight)
                                + fc2_soldier_bias
                            ),
                            name="hero%d_fc2_soldier_1_result_%d" % (hero_index, index),
                        )
                        fc2_soldier_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc2_soldier_result, fc2_soldier_weight_add)
                                + fc2_soldier_bias_add
                            ),
                            name="hero%d_fc2_soldier_1_result_%d_add"
                            % (hero_index, index),
                        )
                        fc3_soldier_result = tf.add(
                            tf.matmul(fc2_soldier_result_add, fc3_soldier_weight_frd),
                            fc3_soldier_bias_frd,
                            name="hero%d_fc3_soldier_1_result_%d" % (hero_index, index),
                        )

                        soldier_1_result_list.append(fc3_soldier_result)
                    soldier_1_concat_result = tf.concat(
                        soldier_1_result_list,
                        axis=1,
                        name="hero%d_soldier_1_concat_result" % hero_index,
                    )

                    reshape_frd_soldier = tf.reshape(
                        soldier_1_concat_result,
                        shape=[-1, 10, 32, 1],
                        name="hero%d_reshape_frd_soldier" % hero_index,
                    )
                    pool_frd_soldier = tf.nn.max_pool(
                        reshape_frd_soldier,
                        [1, 10, 1, 1],
                        [1, 1, 1, 1],
                        padding="VALID",
                        name="hero%d_pool_frd_soldier" % hero_index,
                    )
                    output_dim = int(np.prod(pool_frd_soldier.get_shape()[1:]))
                    reshape_pool_frd_soldier = tf.reshape(
                        pool_frd_soldier,
                        shape=[-1, output_dim],
                        name="hero%d_reshape_pool_frd_soldier" % hero_index,
                    )

                    soldier_2_result_list = []
                    for index in range(len(soldier_11_20)):
                        vec_fc1_input_dim = int(
                            np.prod(soldier_11_20[index].get_shape()[1:])
                        )
                        fc1_soldier_result = tf.nn.relu(
                            (
                                tf.matmul(soldier_11_20[index], fc1_soldier_weight)
                                + fc1_soldier_bias
                            ),
                            name="hero%d_fc1_soldier_2_result_%d" % (hero_index, index),
                        )
                        fc2_soldier_result = tf.nn.relu(
                            (
                                tf.matmul(fc1_soldier_result, fc2_soldier_weight)
                                + fc2_soldier_bias
                            ),
                            name="hero%d_fc2_soldier_2_result_%d" % (hero_index, index),
                        )
                        fc2_soldier_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc2_soldier_result, fc2_soldier_weight_add)
                                + fc2_soldier_bias_add
                            ),
                            name="hero%d_fc2_soldier_2_result_%d_add"
                            % (hero_index, index),
                        )
                        fc3_soldier_result = tf.add(
                            tf.matmul(fc2_soldier_result_add, fc3_soldier_weight_enemy),
                            fc3_soldier_bias_enemy,
                            name="hero%d_fc3_soldier_2_result_%d" % (hero_index, index),
                        )

                        # soldier_emb

                        soldier_2_result_list.append(fc3_soldier_result)
                    soldier_2_concat_result = tf.concat(
                        soldier_2_result_list,
                        axis=1,
                        name="hero%d_soldier_2_concat_result" % hero_index,
                    )

                    reshape_emy_soldier = tf.reshape(
                        soldier_2_concat_result,
                        shape=[-1, 10, 32, 1],
                        name="hero%d_reshape_emy_soldier" % hero_index,
                    )
                    pool_emy_soldier = tf.nn.max_pool(
                        reshape_emy_soldier,
                        [1, 10, 1, 1],
                        [1, 1, 1, 1],
                        padding="VALID",
                        name="hero%d_pool_emy_soldier" % hero_index,
                    )
                    output_dim = int(np.prod(pool_emy_soldier.get_shape()[1:]))
                    reshape_pool_emy_soldier = tf.reshape(
                        pool_emy_soldier,
                        shape=[-1, output_dim],
                        name="hero%d_reshape_pool_emy_soldier" % hero_index,
                    )

                with tf.variable_scope(
                    "hero%d_organ_share" % hero_index, reuse=tf.AUTO_REUSE
                ):
                    organ_1_result_list = []
                    for index in range(len(organ_1_3)):
                        vec_fc1_input_dim = int(
                            np.prod(organ_1_3[index].get_shape()[1:])
                        )
                        fc1_organ_result = tf.nn.relu(
                            (
                                tf.matmul(organ_1_3[index], fc1_organ_weight)
                                + fc1_organ_bias
                            ),
                            name="hero%d_fc1_organ_1_result_%d" % (hero_index, index),
                        )
                        fc2_organ_result = tf.nn.relu(
                            (
                                tf.matmul(fc1_organ_result, fc2_organ_weight)
                                + fc2_organ_bias
                            ),
                            name="hero%d_fc2_organ_1_result_%d" % (hero_index, index),
                        )
                        fc2_organ_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc2_organ_result, fc2_organ_weight_add)
                                + fc2_organ_bias_add
                            ),
                            name="hero%d_fc2_organ_1_result_%d_add"
                            % (hero_index, index),
                        )
                        fc3_organ_result = tf.add(
                            tf.matmul(fc2_organ_result_add, fc3_organ_weight_frd),
                            fc3_organ_bias_frd,
                            name="hero%d_fc3_organ_1_result_%d" % (hero_index, index),
                        )

                        organ_1_result_list.append(fc3_organ_result)
                    organ_1_concat_result = tf.concat(
                        organ_1_result_list,
                        axis=1,
                        name="hero%d_organ_1_concat_result" % hero_index,
                    )

                    reshape_frd_organ = tf.reshape(
                        organ_1_concat_result,
                        shape=[-1, 3, 32, 1],
                        name="hero%d_reshape_frd_organ" % hero_index,
                    )
                    pool_frd_organ = tf.nn.max_pool(
                        reshape_frd_organ,
                        [1, 3, 1, 1],
                        [1, 1, 1, 1],
                        padding="VALID",
                        name="hero%d_pool_frd_organ" % hero_index,
                    )
                    output_dim = int(np.prod(pool_frd_organ.get_shape()[1:]))
                    reshape_pool_frd_organ = tf.reshape(
                        pool_frd_organ,
                        shape=[-1, output_dim],
                        name="hero%d_reshape_pool_frd_organ" % hero_index,
                    )

                    organ_2_result_list = []
                    for index in range(len(organ_4_6)):
                        vec_fc1_input_dim = int(
                            np.prod(organ_4_6[index].get_shape()[1:])
                        )
                        fc1_organ_result = tf.nn.relu(
                            (
                                tf.matmul(organ_4_6[index], fc1_organ_weight)
                                + fc1_organ_bias
                            ),
                            name="hero%d_fc1_organ_2_result_%d" % (hero_index, index),
                        )
                        fc2_organ_result = tf.nn.relu(
                            (
                                tf.matmul(fc1_organ_result, fc2_organ_weight)
                                + fc2_organ_bias
                            ),
                            name="hero%d_fc2_organ_2_result_%d" % (hero_index, index),
                        )
                        fc2_organ_result_add = tf.nn.relu(
                            (
                                tf.matmul(fc2_organ_result, fc2_organ_weight_add)
                                + fc2_organ_bias_add
                            ),
                            name="hero%d_fc2_organ_2_result_%d_add"
                            % (hero_index, index),
                        )
                        fc3_organ_result = tf.add(
                            tf.matmul(fc2_organ_result_add, fc3_organ_weight_enemy),
                            fc3_organ_bias_enemy,
                            name="hero%d_fc3_organ_2_result_%d" % (hero_index, index),
                        )

                        organ_2_result_list.append(fc3_organ_result)
                    organ_2_concat_result = tf.concat(
                        organ_2_result_list,
                        axis=1,
                        name="hero%d_organ_2_concat_result" % hero_index,
                    )

                    reshape_emy_organ = tf.reshape(
                        organ_2_concat_result,
                        shape=[-1, 3, 32, 1],
                        name="hero%d_reshape_emy_organ" % hero_index,
                    )
                    pool_emy_organ = tf.nn.max_pool(
                        reshape_emy_organ,
                        [1, 3, 1, 1],
                        [1, 1, 1, 1],
                        padding="VALID",
                        name="hero%d_pool_emy_organ" % hero_index,
                    )
                    output_dim = int(np.prod(pool_emy_organ.get_shape()[1:]))
                    reshape_pool_emy_organ = tf.reshape(
                        pool_emy_organ,
                        shape=[-1, output_dim],
                        name="hero%d_reshape_pool_emy_organ" % hero_index,
                    )
                    # organ_embedding

                # add none target embedding

                # 3v3 delete spec

                with tf.variable_scope("hero%d_public_concat" % hero_index):
                    # 3v3
                    concat_result = tf.concat(
                        [
                            flatten_conv2_result,
                            reshape_pool_frd_soldier,
                            reshape_pool_emy_soldier,
                            reshape_pool_monster,
                            reshape_pool_frd_organ,
                            reshape_pool_emy_organ,
                            hero_main_concat_result,
                            reshape_pool_hero_frd,
                            reshape_pool_hero_emy,
                            global_info,
                        ],
                        axis=1,
                        name="hero%d_concat_result" % hero_index,
                    )

                with tf.variable_scope(
                    "hero%d_public_weicao" % hero_index, reuse=tf.AUTO_REUSE
                ):
                    public_input_dim = int(np.prod(concat_result.get_shape()[1:]))
                    fc_public_result = tf.nn.relu(
                        (tf.matmul(concat_result, fc_public_weight) + fc_public_bias),
                        name="hero%d_fc_public_result" % hero_index,
                    )
                    this_hero_public_result = tf.split(
                        fc_public_result, [64, 192], axis=1
                    )
                    # shared encoding
                    hero_public_first_result_list.append(this_hero_public_result[0])
                    # individual encoding
                    hero_public_second_result_list.append(this_hero_public_result[1])

        # concat the shared encoding across heros
        with tf.variable_scope("hero_communication"):
            hero_public_concat_result = tf.concat(
                hero_public_first_result_list,
                axis=1,
                name="hero_public_concat_result",
            )
            reshape_hero_public = tf.reshape(
                hero_public_concat_result,
                shape=[-1, 3, 64, 1],
                name="reshape_hero_public",
            )
            pool_hero_public = tf.nn.max_pool(
                reshape_hero_public,
                [1, 3, 1, 1],
                [1, 1, 1, 1],
                padding="VALID",
                name="pool_hero_public",
            )
            hero_public_input_dim = int(np.prod(pool_hero_public.get_shape()[1:]))
            reshape_pool_hero_public = tf.reshape(
                pool_hero_public,
                shape=[-1, hero_public_input_dim],
                name="reshape_pool_hero_public",
            )

        with tf.variable_scope("hero_all_end"):
            fc_public_result_list = []
            for hero_index in range(len(each_hero_data_list)):
                new_fc_public_result_list = []
                new_fc_public_result_list.append(reshape_pool_hero_public)
                new_fc_public_result_list.append(
                    hero_public_second_result_list[hero_index]
                )
                new_fc_public_result = tf.concat(
                    new_fc_public_result_list,
                    axis=1,
                    name="hero%d_new_public_concat_result" % hero_index,
                )
                fc_public_result_list.append(new_fc_public_result)

            # lstm_cell
            # label fc para
            fc1_label_weight_list = []
            fc1_label_bias_list = []
            fc2_label_weight_list = []
            fc2_label_bias_list = []

            for label_index in range(len(self.hero_label_size_list)):
                ### fc1
                fc1_label_weight = self._fc_weight_variable(
                    shape=[256, 64], name="hero_fc1_label_%d_weight" % (label_index)
                )
                fc1_label_bias = self._bias_variable(
                    shape=[64], name="hero_fc1_label_%d_bias" % (label_index)
                )
                fc1_label_weight_list.append(fc1_label_weight)
                fc1_label_bias_list.append(fc1_label_bias)
                ## fc2
                fc2_label_weight = self._fc_weight_variable(
                    shape=[64, self.hero_label_size_list[label_index]],
                    name="hero_fc2_label_%d_weight" % (label_index),
                )
                fc2_label_bias = self._bias_variable(
                    shape=[self.hero_label_size_list[label_index]],
                    name="hero_fc2_label_%d_bias" % (label_index),
                )
                fc2_label_weight_list.append(fc2_label_weight)
                fc2_label_bias_list.append(fc2_label_bias)

            ## value fc para
            fc1_value_weight = self._fc_weight_variable(
                shape=[256, 64], name="hero_fc1_value_weight"
            )
            fc1_value_bias = self._bias_variable(shape=[64], name="hero_fc1_value_bias")
            fc2_value_weight_list = []
            fc2_value_bias_list = []
            for value_index in range(self.value_head_num):
                fc2_value_weight = self._fc_weight_variable(
                    shape=[64, 1],
                    name="hero_fc2_value_%d_weight" % (value_index),
                    trainable=True,
                )
                fc2_value_bias = self._bias_variable(
                    shape=[1],
                    name="hero_fc2_value_%d_bias" % (value_index),
                    trainable=True,
                )
                fc2_value_weight_list.append(fc2_value_weight)
                fc2_value_bias_list.append(fc2_value_bias)

        for hero_index in range(len(each_hero_data_list)):
            this_hero_fc_result_list = []
            this_hero_variable_list = []
            ## action layer ###
            with tf.variable_scope("hero%d_out" % hero_index):
                for label_index in range(len(self.hero_label_size_list)):
                    with tf.variable_scope(
                        "hero%d_fc1_label_%d" % (hero_index, label_index)
                    ):
                        fc1_label_result = tf.nn.relu(
                            (
                                tf.matmul(
                                    fc_public_result_list[hero_index],
                                    fc1_label_weight_list[label_index],
                                )
                                + fc1_label_bias_list[label_index]
                            ),
                            name="hero%d_fc1_label_%d_result"
                            % (hero_index, label_index),
                        )

                    with tf.variable_scope(
                        "hero%d_fc2_label_%d" % (hero_index, label_index)
                    ):
                        fc2_label_result = tf.add(
                            tf.matmul(
                                fc1_label_result, fc2_label_weight_list[label_index]
                            ),
                            fc2_label_bias_list[label_index],
                            name="hero%d_fc2_label_%d_result"
                            % (hero_index, label_index),
                        )
                        this_hero_fc_result_list.append(fc2_label_result)

                with tf.variable_scope("hero%d_fc1_value" % hero_index):
                    fc1_value_result = tf.nn.relu(
                        (
                            tf.matmul(
                                fc_public_result_list[hero_index], fc1_value_weight
                            )
                            + fc1_value_bias
                        ),
                        name="hero%d_fc1_value_result" % hero_index,
                    )
                for value_index in range(self.value_head_num):
                    with tf.variable_scope(
                        "hero%d_fc2_value_%d" % (hero_index, value_index)
                    ):
                        fc2_value_result = tf.add(
                            tf.matmul(
                                fc1_value_result, fc2_value_weight_list[value_index]
                            ),
                            fc2_value_bias_list[value_index],
                            name="hero%d_fc2_value_%d_result"
                            % (hero_index, value_index),
                        )
                        this_hero_fc_result_list.append(fc2_value_result)
                each_hero_fc_result_list.append(this_hero_fc_result_list)

                # for saving this hero network
                this_hero_str = "hero%d" % hero_index
                for this_hero_variable in tf.trainable_variables():
                    if this_hero_str in this_hero_variable.name:
                        this_hero_variable_list.append(this_hero_variable)
                self.restore_list.append(this_hero_variable_list)

        return each_hero_fc_result_list

    def _conv_weight_variable(self, shape, name, trainable=True):
        # initializer = tf.contrib.layers.xavier_initializer_conv2d()
        initializer = tf.orthogonal_initializer()
        return tf.get_variable(
            name, shape=shape, initializer=initializer, trainable=trainable
        )

    def _fc_weight_variable(self, shape, name, trainable=True):
        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.orthogonal_initializer()
        return tf.get_variable(
            name, shape=shape, initializer=initializer, trainable=trainable
        )

    def _bias_variable(self, shape, name, trainable=True):
        initializer = tf.constant_initializer(0.0)
        return tf.get_variable(
            name, shape=shape, initializer=initializer, trainable=trainable
        )
