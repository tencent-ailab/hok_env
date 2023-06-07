import numpy as np
import tensorflow as tf
from common_config import DimConfig
from common_config import ModelConfig as Config


class Algorithm:
    def __init__(self):
        # feature configure parameter
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        # self.need_reinforce_param_button_label_list = Config.NEED_REINFORCE_PARAM_BUTTON_LABEL_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.batch_size = Config.BATCH_SIZE * 16
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]

    def get_input_tensors(self):
        return [
            self.feature_ph,
            self.legal_action_ph,
            self.lstm_cell_ph,
            self.lstm_hidden_ph,
        ]

    def get_output_tensors(self):
        return [self.logits, self.value, self.lstm_cell_output, self.lstm_hidden_output]
        # + [self.used_legal_action]

    def build_infer_graph(self):
        if self.graph is not None:
            return self.graph
        self.graph = self._build_infer_graph()
        return self.graph

    def _build_infer_graph(self):
        self.graph = tf.Graph()
        # cpu_num = 1
        # config = tf.ConfigProto(device_count={"CPU": cpu_num},
        #                         inter_op_parallelism_threads=cpu_num,
        #                         intra_op_parallelism_threads=cpu_num,
        #                         log_device_placement=False)
        with self.graph.as_default():
            # Build input placeholders
            self.feature_ph = tf.placeholder(
                shape=(self.batch_size, self.feature_dim),
                name="feature",
                dtype=np.float32,
            )
            self.legal_action_ph = tf.placeholder(
                shape=(self.batch_size, self.legal_action_dim),
                name="legal_action",
                dtype=np.float32,
            )
            self.lstm_cell_ph = tf.placeholder(
                shape=(self.batch_size, self.lstm_hidden_dim),
                name="lstm_cell",
                dtype=np.float32,
            )
            self.lstm_hidden_ph = tf.placeholder(
                shape=(self.batch_size, self.lstm_hidden_dim),
                name="lstm_hidden",
                dtype=np.float32,
            )
            print(
                "Build net: ",
                self.feature_dim,
                self.legal_action_dim,
                self.lstm_hidden_dim,
            )

            # build graph (outputs)
            # data_list = tf.split(datas, self.cut_points, axis=1)
            feature = tf.reshape(self.feature_ph, [-1, self.seri_vec_split_shape[0][0]])
            legal_action = tf.reshape(
                self.legal_action_ph, [-1, np.sum(self.legal_action_size)]
            )
            seri_vec = (feature, legal_action)
            # seri_vec = tf.reshape(self.feature_ph, [-1, self.data_split_shape[0]])
            init_lstm_cell, init_lstm_hidden = self.lstm_cell_ph, self.lstm_hidden_ph
            fc_label_result_list = self._inference(
                seri_vec, init_lstm_cell, init_lstm_hidden, only_inference=True
            )
            logits_list, value_list = (
                fc_label_result_list[:-1],
                fc_label_result_list[-1],
            )
            self.logits = tf.layers.flatten(tf.concat(logits_list, axis=1))
            self.value = tf.layers.flatten(value_list[0])
            # self.init_saver = tf.train.Saver(tf.global_variables())
            self.init = tf.global_variables_initializer()
            # self.sess = tf.Session(config=config)
            # self.sess.run(tf.global_variables_initializer())
        return self.graph

    def build_graph(self, datas, update):
        # add split datas
        data_list = tf.split(datas, self.cut_points, axis=1)
        #  the meaning of each data in data_list should be as the same as that in GpuProxy.py
        for i, data in enumerate(data_list):
            data = tf.reshape(data, [-1])
            data_list[i] = tf.cast(data, dtype=tf.float32)
        seri_vec = data_list[0]
        seri_vec = tf.reshape(seri_vec, [-1, self.data_split_shape[0]])

        reward = data_list[1]
        reward = tf.reshape(reward, [-1, self.data_split_shape[1]])

        advantage = data_list[2]
        advantage = tf.reshape(advantage, [-1, self.data_split_shape[2]])

        label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            # label_list[shape_index] = tf.cast(label_list,dtype=tf.int32)
            label_list[shape_index] = tf.cast(
                tf.reshape(
                    label_list[shape_index],
                    [-1, self.data_split_shape[3 + shape_index]],
                ),
                dtype=tf.int32,
            )

        squeeze_label_list = []
        for ele in label_list:
            squeeze_label_list.append(tf.squeeze(ele, axis=[1]))

        old_label_probability_list = data_list[
            3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)
        ]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = tf.reshape(
                old_label_probability_list[shape_index],
                [
                    -1,
                    self.data_split_shape[3 + len(self.label_size_list) + shape_index],
                ],
            )

        weight_list = data_list[
            3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)
        ]
        for shape_index in range(len(self.label_size_list)):
            weight_list[shape_index] = tf.reshape(
                weight_list[shape_index],
                [
                    -1,
                    self.data_split_shape[
                        3 + 2 * len(self.label_size_list) + shape_index
                    ],
                ],
            )

        is_train = data_list[-3]
        is_train = tf.reshape(is_train, [-1, self.data_split_shape[-3]])

        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]
        # build network
        fc_label_result_list = self._inference(
            seri_vec,
            init_lstm_cell,
            init_lstm_hidden,
        )
        # calculate loss
        loss = self._calculate_loss(
            label_list,
            old_label_probability_list,
            fc_label_result_list[:-1],
            reward,
            advantage,
            fc_label_result_list[-1],
            seri_vec,
            is_train,
            weight_list,
        )
        info_list = {
            "loss": loss,
            "value_cost": self.value_cost,
            "entropy_cost": self.entropy_cost,
            "policy_cost": self.policy_cost,
        }

        return loss, info_list

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.00001)

    def _squeeze_tensor(
        self,
        unsqueeze_reward,
        unsqueeze_advantage,
        unsqueeze_label_list,
        unsqueeze_frame_is_train,
        unsqueeze_weight_list,
    ):
        reward = tf.squeeze(unsqueeze_reward, axis=[1])
        advantage = tf.squeeze(unsqueeze_advantage, axis=[1])
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(tf.squeeze(ele, axis=[1]))
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(tf.squeeze(weight, axis=[1]))
        frame_is_train = tf.squeeze(unsqueeze_frame_is_train, axis=[1])
        return reward, advantage, label_list, frame_is_train, weight_list

    def _calculate_loss(
        self,
        unsqueeze_label_list,
        old_label_probability_list,
        fc2_label_list,
        unsqueeze_reward,
        unsqueeze_advantage,
        fc2_value_result,
        seri_vec,
        unsqueeze_is_train,
        unsqueeze_weight_list,
    ):

        reward, advantage, label_list, _, weight_list = self._squeeze_tensor(
            unsqueeze_reward,
            unsqueeze_advantage,
            unsqueeze_label_list,
            unsqueeze_is_train,
            unsqueeze_weight_list,
        )
        _, split_feature_legal_action = tf.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            axis=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = tf.reshape(
            split_feature_legal_action, feature_legal_action_shape
        )
        # self.feature_legal_action = feature_legal_action

        legal_action_flag_list = tf.split(
            feature_legal_action, self.label_size_list, axis=1
        )

        # loss of value net
        fc2_value_result_squeezed = tf.squeeze(fc2_value_result, axis=[1])
        self.value_cost = 0.5 * tf.reduce_mean(
            tf.square(reward - fc2_value_result_squeezed), axis=0
        )
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * tf.reduce_mean(tf.square(new_advantage), axis=0)

        # for entropy loss calculate
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []
        # policy loss: ppo clip loss
        self.policy_cost = tf.constant(0.0, dtype=tf.float32)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = tf.constant(0.0, dtype=tf.float32)
                one_hot_actions = tf.one_hot(
                    label_list[task_index], self.label_size_list[task_index]
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
                policy_log_p = tf.log(policy_p + 0.00001)
                old_policy_p = tf.reduce_sum(
                    one_hot_actions * old_label_probability_list[task_index] + 0.00001,
                    axis=1,
                )
                old_policy_log_p = tf.log(old_policy_p)
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
                temp_policy_loss = -tf.reduce_sum(
                    tf.to_float(weight_list[task_index]) * tf.minimum(surr1, surr2)
                ) / tf.maximum(tf.reduce_sum(tf.to_float(weight_list[task_index])), 1.0)
                self.policy_cost = self.policy_cost + temp_policy_loss
        # cross entropy loss
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -tf.reduce_sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * tf.log(
                        label_probability_list[current_entropy_loss_index] + 0.00001
                    ),
                    axis=1,
                )
                temp_entropy_loss = -tf.reduce_sum(
                    (temp_entropy_loss * tf.to_float(weight_list[task_index]))
                ) / tf.maximum(
                    tf.reduce_sum(tf.to_float(weight_list[task_index])), 1.0
                )  # add - because need to minize
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = tf.constant(0.0, dtype=tf.float32)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = tf.constant(0.0, dtype=tf.float32)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element
        self.entropy_cost_list = entropy_loss_list
        # sum all type cost
        self.cost_all = (
            self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        )
        # make output information
        # add loss information
        self.all_loss_list = [
            self.cost_all,
            self.value_cost,
            self.policy_cost,
            self.entropy_cost,
        ]
        return self.cost_all

    def _inference(
        self, seri_vec, init_lstm_cell, init_lstm_hidden, only_inference=False
    ):

        # model design
        if only_inference:
            # actor input seri_vec as (feature, legal_action)
            split_feature_vec, split_feature_legal_action = seri_vec
        else:
            split_feature_vec, split_feature_legal_action = tf.split(
                seri_vec,
                [
                    np.prod(self.seri_vec_split_shape[0]),
                    np.prod(self.seri_vec_split_shape[1]),
                ],
                axis=1,
            )
        feature_vec_shape = list(self.seri_vec_split_shape[0])
        # feature_vec_shape.insert(0, -1)
        feature_vec_shape.insert(0, self.batch_size)
        feature_vec = tf.reshape(split_feature_vec, feature_vec_shape)
        feature_vec = tf.identity(feature_vec, name="feature_vec")

        lstm_cell_state = tf.reshape(init_lstm_cell, [-1, self.lstm_unit_size])
        lstm_hidden_state = tf.reshape(init_lstm_hidden, [-1, self.lstm_unit_size])
        lstm_initial_state = tf.nn.rnn_cell.LSTMStateTuple(
            lstm_cell_state, lstm_hidden_state
        )

        result_list = []

        hero_dim = (
            int(np.sum(DimConfig.DIM_OF_HERO_FRD))
            + int(np.sum(DimConfig.DIM_OF_HERO_EMY))
            + int(np.sum(DimConfig.DIM_OF_HERO_MAIN))
        )
        soldier_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)) + int(
            np.sum(DimConfig.DIM_OF_SOLDIER_11_20)
        )
        organ_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1_2)) + int(
            np.sum(DimConfig.DIM_OF_ORGAN_3_4)
        )
        global_info_dim = int(np.sum(DimConfig.DIM_OF_GLOBAL_INFO))

        with tf.variable_scope("feature_vec_split"):
            feature_vec_split_list = tf.split(
                feature_vec, [hero_dim, soldier_dim, organ_dim, global_info_dim], axis=1
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
                    int(np.sum(DimConfig.DIM_OF_ORGAN_1_2)),
                    int(np.sum(DimConfig.DIM_OF_ORGAN_3_4)),
                ],
                axis=1,
            )
            global_info_list = feature_vec_split_list[3]

            soldier_1_10 = tf.split(
                soldier_vec_list[0], DimConfig.DIM_OF_SOLDIER_1_10, axis=1
            )
            soldier_11_20 = tf.split(
                soldier_vec_list[1], DimConfig.DIM_OF_SOLDIER_11_20, axis=1
            )
            organ_1_2 = tf.split(organ_vec_list[0], DimConfig.DIM_OF_ORGAN_1_2, axis=1)
            organ_3_4 = tf.split(organ_vec_list[1], DimConfig.DIM_OF_ORGAN_3_4, axis=1)
            hero_frd = tf.split(hero_vec_list[0], DimConfig.DIM_OF_HERO_FRD, axis=1)
            hero_emy = tf.split(hero_vec_list[1], DimConfig.DIM_OF_HERO_EMY, axis=1)
            hero_main = tf.split(hero_vec_list[2], DimConfig.DIM_OF_HERO_MAIN, axis=1)
            global_info = global_info_list

        tar_embed_list = []
        # non_target_embedding
        tar_embed_list.append(
            tf.constant(0.1, shape=[self.batch_size, self.target_embed_dim])
        )

        with tf.variable_scope("hero_main"):

            for index in range(len(hero_main)):
                vec_fc1_input_dim = int(np.prod(hero_main[index].get_shape()[1:]))
                fc1_hero_weight = self._fc_weight_variable(
                    shape=[vec_fc1_input_dim, 64], name="fc1_hero_main_weight"
                )
                fc1_hero_bias = self._bias_variable(
                    shape=[64], name="fc1_hero_main_bias"
                )
                fc1_hero_result = tf.nn.relu(
                    (tf.matmul(hero_main[index], fc1_hero_weight) + fc1_hero_bias),
                    name="fc1_hero_main_result_%d" % index,
                )

                fc2_hero_weight = self._fc_weight_variable(
                    shape=[64, 32], name="fc2_hero_main_weight"
                )
                fc2_hero_bias = self._bias_variable(
                    shape=[32], name="fc2_hero_main_bias"
                )
                fc2_hero_result = tf.nn.relu(
                    (tf.matmul(fc1_hero_result, fc2_hero_weight) + fc2_hero_bias),
                    name="fc2_hero_main_result_%d" % index,
                )

                fc3_hero_weight = self._fc_weight_variable(
                    shape=[32, 16], name="fc3_hero_main_weight"
                )
                fc3_hero_bias = self._bias_variable(
                    shape=[16], name="fc3_hero_main_bias"
                )
                fc3_hero_result = tf.add(
                    tf.matmul(fc2_hero_result, fc3_hero_weight),
                    fc3_hero_bias,
                    name="fc3_hero_main_result_%d" % index,
                )

            hero_main_concat_result = fc3_hero_result

        with tf.variable_scope("hero_share", reuse=tf.AUTO_REUSE):
            # with tf.variable_scope("hero_emy"):
            hero_emy_result_list = []
            for index in range(len(hero_emy)):
                vec_fc1_input_dim = int(np.prod(hero_emy[index].get_shape()[1:]))
                fc1_hero_weight = self._fc_weight_variable(
                    shape=[vec_fc1_input_dim, 512], name="fc1_hero_weight"
                )
                fc1_hero_bias = self._bias_variable(shape=[512], name="fc1_hero_bias")
                fc1_hero_result = tf.nn.relu(
                    (tf.matmul(hero_emy[index], fc1_hero_weight) + fc1_hero_bias),
                    name="fc1_hero_emy_result_%d" % index,
                )

                fc2_hero_weight = self._fc_weight_variable(
                    shape=[512, 256], name="fc2_hero_weight"
                )
                fc2_hero_bias = self._bias_variable(shape=[256], name="fc2_hero_bias")
                fc2_hero_result = tf.nn.relu(
                    (tf.matmul(fc1_hero_result, fc2_hero_weight) + fc2_hero_bias),
                    name="fc2_hero_emy_result_%d" % index,
                )

                fc3_hero_weight = self._fc_weight_variable(
                    shape=[256, 128], name="fc3_hero_emy_weight"
                )
                fc3_hero_bias = self._bias_variable(
                    shape=[128], name="fc3_hero_emy_bias"
                )
                fc3_hero_result = tf.add(
                    tf.matmul(fc2_hero_result, fc3_hero_weight),
                    fc3_hero_bias,
                    name="fc3_hero_emy_result_%d" % index,
                )
                # emy_hero_embedding
                _, split_1 = tf.split(fc3_hero_result, [96, 32], axis=1)
                tar_embed_list.append(split_1)

                hero_emy_result_list.append(fc3_hero_result)
            hero_emy_concat_result = tf.concat(
                hero_emy_result_list, axis=1, name="hero_emy_concat_result"
            )

            reshape_hero_emy = tf.reshape(
                hero_emy_concat_result, shape=[-1, 1, 128, 1], name="reshape_hero_emy"
            )
            pool_hero_emy = tf.nn.max_pool(
                reshape_hero_emy,
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                padding="VALID",
                name="pool_hero_emy",
            )
            output_dim = int(np.prod(pool_hero_emy.get_shape()[1:]))
            reshape_pool_hero_emy = tf.reshape(
                pool_hero_emy, shape=[-1, output_dim], name="reshape_pool_hero_emy"
            )

            hero_frd_result_list = []
            for index in range(len(hero_frd)):
                vec_fc1_input_dim = int(np.prod(hero_frd[index].get_shape()[1:]))
                fc1_hero_weight = self._fc_weight_variable(
                    shape=[vec_fc1_input_dim, 512], name="fc1_hero_weight"
                )
                fc1_hero_bias = self._bias_variable(shape=[512], name="fc1_hero_bias")
                fc1_hero_result = tf.nn.relu(
                    (tf.matmul(hero_frd[index], fc1_hero_weight) + fc1_hero_bias),
                    name="fc1_hero_frd_result_%d" % index,
                )

                fc2_hero_weight = self._fc_weight_variable(
                    shape=[512, 256], name="fc2_hero_weight"
                )
                fc2_hero_bias = self._bias_variable(shape=[256], name="fc2_hero_bias")
                fc2_hero_result = tf.nn.relu(
                    (tf.matmul(fc1_hero_result, fc2_hero_weight) + fc2_hero_bias),
                    name="fc2_hero_frd_result_%d" % index,
                )

                fc3_hero_weight = self._fc_weight_variable(
                    shape=[256, 128], name="fc3_hero_frd_weight"
                )
                fc3_hero_bias = self._bias_variable(
                    shape=[128], name="fc3_hero_frd_bias"
                )
                fc3_hero_result = tf.add(
                    tf.matmul(fc2_hero_result, fc3_hero_weight),
                    fc3_hero_bias,
                    name="fc3_hero_frd_result_%d" % index,
                )
                #  frd_hero_embedding
                split_0, split_1 = tf.split(fc3_hero_result, [96, 32], axis=1)
                tar_embed_list.append(split_1)

                hero_frd_result_list.append(fc3_hero_result)
            hero_frd_concat_result = tf.concat(
                hero_frd_result_list, axis=1, name="hero_frd_concat_result"
            )

            reshape_hero_frd = tf.reshape(
                hero_frd_concat_result, shape=[-1, 1, 128, 1], name="reshape_hero_frd"
            )
            pool_hero_frd = tf.nn.max_pool(
                reshape_hero_frd,
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                padding="VALID",
                name="pool_hero_frd",
            )
            output_dim = int(np.prod(pool_hero_frd.get_shape()[1:]))
            reshape_pool_hero_frd = tf.reshape(
                pool_hero_frd, shape=[-1, output_dim], name="reshape_pool_hero_frd"
            )

        with tf.variable_scope("soldier_share", reuse=tf.AUTO_REUSE):
            soldier_1_result_list = []
            for index in range(len(soldier_1_10)):
                vec_fc1_input_dim = int(np.prod(soldier_1_10[index].get_shape()[1:]))
                fc1_soldier_weight = self._fc_weight_variable(
                    shape=[vec_fc1_input_dim, 64], name="fc1_soldier_weight"
                )
                fc1_soldier_bias = self._bias_variable(
                    shape=[64], name="fc1_soldier_bias"
                )
                fc1_soldier_result = tf.nn.relu(
                    (
                        tf.matmul(soldier_1_10[index], fc1_soldier_weight)
                        + fc1_soldier_bias
                    ),
                    name="fc1_soldier_1_result_%d" % index,
                )

                fc2_soldier_weight = self._fc_weight_variable(
                    shape=[64, 64], name="fc2_soldier_weight"
                )
                fc2_soldier_bias = self._bias_variable(
                    shape=[64], name="fc2_soldier_bias"
                )
                fc2_soldier_result = tf.nn.relu(
                    (
                        tf.matmul(fc1_soldier_result, fc2_soldier_weight)
                        + fc2_soldier_bias
                    ),
                    name="fc2_soldier_1_result_%d" % index,
                )

                fc3_soldier_weight = self._fc_weight_variable(
                    shape=[64, 32], name="fc3_soldier_1_weight"
                )
                fc3_soldier_bias = self._bias_variable(
                    shape=[32], name="fc3_soldier_1_bias"
                )
                fc3_soldier_result = tf.add(
                    tf.matmul(fc2_soldier_result, fc3_soldier_weight),
                    fc3_soldier_bias,
                    name="fc3_soldier_1_result_%d" % index,
                )

                soldier_1_result_list.append(fc3_soldier_result)
            soldier_1_concat_result = tf.concat(
                soldier_1_result_list, axis=1, name="soldier_1_concat_result"
            )

            reshape_frd_soldier = tf.reshape(
                soldier_1_concat_result,
                shape=[-1, 4, 32, 1],
                name="reshape_frd_soldier",
            )
            pool_frd_soldier = tf.nn.max_pool(
                reshape_frd_soldier,
                [1, 4, 1, 1],
                [1, 1, 1, 1],
                padding="VALID",
                name="pool_frd_soldier",
            )
            output_dim = int(np.prod(pool_frd_soldier.get_shape()[1:]))
            reshape_pool_frd_soldier = tf.reshape(
                pool_frd_soldier,
                shape=[-1, output_dim],
                name="reshape_pool_frd_soldier",
            )

            # with tf.variable_scope("soldier_31_60"):
            soldier_2_result_list = []
            for index in range(len(soldier_11_20)):
                vec_fc1_input_dim = int(np.prod(soldier_11_20[index].get_shape()[1:]))
                fc1_soldier_weight = self._fc_weight_variable(
                    shape=[vec_fc1_input_dim, 64], name="fc1_soldier_weight"
                )
                fc1_soldier_bias = self._bias_variable(
                    shape=[64], name="fc1_soldier_bias"
                )
                fc1_soldier_result = tf.nn.relu(
                    (
                        tf.matmul(soldier_11_20[index], fc1_soldier_weight)
                        + fc1_soldier_bias
                    ),
                    name="fc1_soldier_2_result_%d" % index,
                )

                fc2_soldier_weight = self._fc_weight_variable(
                    shape=[64, 64], name="fc2_soldier_weight"
                )
                fc2_soldier_bias = self._bias_variable(
                    shape=[64], name="fc2_soldier_bias"
                )
                fc2_soldier_result = tf.nn.relu(
                    (
                        tf.matmul(fc1_soldier_result, fc2_soldier_weight)
                        + fc2_soldier_bias
                    ),
                    name="fc2_soldier_2_result_%d" % index,
                )

                fc3_soldier_weight = self._fc_weight_variable(
                    shape=[64, 32], name="fc3_soldier_2_weight"
                )
                fc3_soldier_bias = self._bias_variable(
                    shape=[32], name="fc3_soldier_2_bias"
                )
                fc3_soldier_result = tf.add(
                    tf.matmul(fc2_soldier_result, fc3_soldier_weight),
                    fc3_soldier_bias,
                    name="fc3_soldier_2_result_%d" % index,
                )
                #  emy soldier embedding
                # split_0, split_1 = tf.split(fc3_soldier_result, [16,16], axis=1)
                tar_embed_list.append(fc3_soldier_result)

                soldier_2_result_list.append(fc3_soldier_result)
            soldier_2_concat_result = tf.concat(
                soldier_2_result_list, axis=1, name="soldier_2_concat_result"
            )

            reshape_emy_soldier = tf.reshape(
                soldier_2_concat_result,
                shape=[-1, 4, 32, 1],
                name="reshape_emy_soldier",
            )
            pool_emy_soldier = tf.nn.max_pool(
                reshape_emy_soldier,
                [1, 4, 1, 1],
                [1, 1, 1, 1],
                padding="VALID",
                name="pool_emy_soldier",
            )
            output_dim = int(np.prod(pool_emy_soldier.get_shape()[1:]))
            reshape_pool_emy_soldier = tf.reshape(
                pool_emy_soldier,
                shape=[-1, output_dim],
                name="reshape_pool_emy_soldier",
            )

        with tf.variable_scope("organ_share", reuse=tf.AUTO_REUSE):
            organ_1_result_list = []
            for index in range(len(organ_1_2)):
                vec_fc1_input_dim = int(np.prod(organ_1_2[index].get_shape()[1:]))
                fc1_organ_weight = self._fc_weight_variable(
                    shape=[vec_fc1_input_dim, 64], name="fc1_organ_weight"
                )
                fc1_organ_bias = self._bias_variable(shape=[64], name="fc1_organ_bias")
                fc1_organ_result = tf.nn.relu(
                    (tf.matmul(organ_1_2[index], fc1_organ_weight) + fc1_organ_bias),
                    name="fc1_organ_1_result_%d" % index,
                )

                fc2_organ_weight = self._fc_weight_variable(
                    shape=[64, 64], name="fc2_organ_weight"
                )
                fc2_organ_bias = self._bias_variable(shape=[64], name="fc2_organ_bias")
                fc2_organ_result = tf.nn.relu(
                    (tf.matmul(fc1_organ_result, fc2_organ_weight) + fc2_organ_bias),
                    name="fc2_organ_1_result_%d" % index,
                )

                fc3_organ_weight = self._fc_weight_variable(
                    shape=[64, 32], name="fc3_organ_1_weight"
                )
                fc3_organ_bias = self._bias_variable(
                    shape=[32], name="fc3_organ_1_bias"
                )
                fc3_organ_result = tf.add(
                    tf.matmul(fc2_organ_result, fc3_organ_weight),
                    fc3_organ_bias,
                    name="fc3_organ_1_result_%d" % index,
                )

                organ_1_result_list.append(fc3_organ_result)
            organ_1_concat_result = tf.concat(
                organ_1_result_list, axis=1, name="organ_1_concat_result"
            )

            reshape_frd_organ = tf.reshape(
                organ_1_concat_result, shape=[-1, 2, 32, 1], name="reshape_frd_organ"
            )
            pool_frd_organ = tf.nn.max_pool(
                reshape_frd_organ,
                [1, 2, 1, 1],
                [1, 1, 1, 1],
                padding="VALID",
                name="pool_frd_organ",
            )
            output_dim = int(np.prod(pool_frd_organ.get_shape()[1:]))
            reshape_pool_frd_organ = tf.reshape(
                pool_frd_organ, shape=[-1, output_dim], name="reshape_pool_frd_organ"
            )

            organ_2_result_list = []
            for index in range(len(organ_3_4)):
                vec_fc1_input_dim = int(np.prod(organ_3_4[index].get_shape()[1:]))
                fc1_organ_weight = self._fc_weight_variable(
                    shape=[vec_fc1_input_dim, 64], name="fc1_organ_weight"
                )
                fc1_organ_bias = self._bias_variable(shape=[64], name="fc1_organ_bias")
                fc1_organ_result = tf.nn.relu(
                    (tf.matmul(organ_3_4[index], fc1_organ_weight) + fc1_organ_bias),
                    name="fc1_organ_2_result_%d" % index,
                )

                fc2_organ_weight = self._fc_weight_variable(
                    shape=[64, 64], name="fc2_organ_weight"
                )
                fc2_organ_bias = self._bias_variable(shape=[64], name="fc2_organ_bias")
                fc2_organ_result = tf.nn.relu(
                    (tf.matmul(fc1_organ_result, fc2_organ_weight) + fc2_organ_bias),
                    name="fc2_organ_2_result_%d" % index,
                )

                fc3_organ_weight = self._fc_weight_variable(
                    shape=[64, 32], name="fc3_organ_2_weight"
                )
                fc3_organ_bias = self._bias_variable(
                    shape=[32], name="fc3_organ_2_bias"
                )
                fc3_organ_result = tf.add(
                    tf.matmul(fc2_organ_result, fc3_organ_weight),
                    fc3_organ_bias,
                    name="fc3_organ_2_result_%d" % index,
                )

                organ_2_result_list.append(fc3_organ_result)
            organ_2_concat_result = tf.concat(
                organ_2_result_list, axis=1, name="organ_2_concat_result"
            )

            reshape_emy_organ = tf.reshape(
                organ_2_concat_result, shape=[-1, 2, 32, 1], name="reshape_emy_organ"
            )
            pool_emy_organ = tf.nn.max_pool(
                reshape_emy_organ,
                [1, 2, 1, 1],
                [1, 1, 1, 1],
                padding="VALID",
                name="pool_emy_organ",
            )
            output_dim = int(np.prod(pool_emy_organ.get_shape()[1:]))
            reshape_pool_emy_organ = tf.reshape(
                pool_emy_organ, shape=[-1, output_dim], name="reshape_pool_emy_organ"
            )
            #  emy_organ_embedding
            # split_0, split_1 = tf.split(reshape_pool_emy_organ, [16,16], axis=1)
            tar_embed_list.append(reshape_pool_emy_organ)

        with tf.variable_scope("public_concat"):
            concat_result = tf.concat(
                [
                    reshape_pool_frd_soldier,
                    reshape_pool_emy_soldier,
                    reshape_pool_frd_organ,
                    reshape_pool_emy_organ,
                    hero_main_concat_result,
                    reshape_pool_hero_frd,
                    reshape_pool_hero_emy,
                    global_info,
                ],
                axis=1,
                name="concat_result",
            )
        with tf.variable_scope("public_weicao"):
            public_input_dim = int(np.prod(concat_result.get_shape()[1:]))
            fc_public_weight = self._fc_weight_variable(
                shape=[public_input_dim, 512], name="fc_public_weight"
            )
            fc_public_bias = self._bias_variable(shape=[512], name="fc_public_bias")
            fc_public_result = tf.nn.relu(
                (tf.matmul(concat_result, fc_public_weight) + fc_public_bias),
                name="fc_public_result",
            )
            # for lstm
            reshape_fc_public_result = tf.reshape(
                fc_public_result,
                [-1, self.lstm_time_steps, 512],
                name="reshape_fc_public_result",
            )

        with tf.variable_scope("public_lstm"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.lstm_unit_size, forget_bias=1.0
            )
            with tf.variable_scope("rnn"):
                state = lstm_initial_state
                lstm_output_list = []
                for step in range(self.lstm_time_steps):
                    lstm_output, state = lstm_cell(
                        reshape_fc_public_result[:, step, :], state
                    )
                    lstm_output_list.append(lstm_output)
                lstm_outputs = tf.concat(lstm_output_list, axis=1, name="lstm_outputs")
                self.lstm_cell_output = state.c
                self.lstm_hidden_output = state.h
            reshape_lstm_outputs_result = tf.reshape(
                lstm_outputs,
                [-1, self.lstm_unit_size],
                name="reshape_lstm_outputs_result",
            )

        #  action layer # #
        for index in range(0, len(self.label_size_list) - 1):
            with tf.variable_scope("fc2_label_%d" % (index)):
                fc2_label_weight = self._fc_weight_variable(
                    shape=[self.lstm_unit_size, self.label_size_list[index]],
                    name="fc2_label_%d_weight" % (index),
                )
                fc2_label_bias = self._bias_variable(
                    shape=[self.label_size_list[index]],
                    name="fc2_label_%d_bias" % (index),
                )
                fc2_label_result = tf.add(
                    tf.matmul(reshape_lstm_outputs_result, fc2_label_weight),
                    fc2_label_bias,
                    name="fc2_label_%d_result" % (index),
                )
                result_list.append(fc2_label_result)

        with tf.variable_scope("fc2_label_%d" % (len(self.label_size_list) - 1)):
            fc2_label_weight = self._fc_weight_variable(
                shape=[self.lstm_unit_size, self.target_embed_dim],
                name="fc2_label_%d_weight" % (len(self.label_size_list) - 1),
            )
            fc2_label_bias = self._bias_variable(
                shape=[self.target_embed_dim],
                name="fc2_label_%d_bias" % (len(self.label_size_list) - 1),
            )
            fc2_label_result = tf.add(
                tf.matmul(reshape_lstm_outputs_result, fc2_label_weight),
                fc2_label_bias,
                name="fc2_label_%d_result" % (len(self.label_size_list) - 1),
            )

            for t in tar_embed_list:
                print("t shape", t.shape)

            tar_embedding = tf.stack(
                tar_embed_list, axis=1
            )  # (Batch_size, unit_num, embed)
            ulti_tar_embedding = tf.layers.dense(
                tar_embedding, self.target_embed_dim, use_bias=False
            )  # (Batch_size, unit_num,embed)
            reshape_fc2_label_result = tf.reshape(
                fc2_label_result,
                shape=[-1, self.target_embed_dim, 1],
                name="reshape_target_embed",
            )
            fc3_label_result = tf.matmul(
                ulti_tar_embedding, reshape_fc2_label_result
            )  # (Batch_size,unit_num,1)
            target_output_dim = int(np.prod(fc3_label_result.get_shape()[1:]))
            reshape_fc3_label_result = tf.reshape(
                fc3_label_result, shape=[-1, target_output_dim], name="target_output"
            )
            result_list.append(reshape_fc3_label_result)

        with tf.variable_scope("fc1_value"):
            fc1_value_weight = self._fc_weight_variable(
                shape=[self.lstm_unit_size, 64], name="fc1_value_weight"
            )
            fc1_value_bias = self._bias_variable(shape=[64], name="fc1_value_bias")
            fc1_value_result = tf.nn.relu(
                (
                    tf.matmul(reshape_lstm_outputs_result, fc1_value_weight)
                    + fc1_value_bias
                ),
                name="fc1_value_result",
            )

        with tf.variable_scope("fc2_value"):
            fc2_value_weight = self._fc_weight_variable(
                shape=[64, 1], name="fc2_value_weight"
            )
            fc2_value_bias = self._bias_variable(shape=[1], name="fc2_value_bias")
            fc2_value_result = tf.add(
                tf.matmul(fc1_value_result, fc2_value_weight),
                fc2_value_bias,
                name="fc2_value_result",
            )
            result_list.append(fc2_value_result)
        return result_list

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

    def _embed_variable(self, shape, name, trainable=True):
        initializer = tf.orthogonal_initializer()
        return tf.get_variable(
            name, shape=shape, initializer=initializer, trainable=trainable
        )
