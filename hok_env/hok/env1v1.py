# -*- coding: utf-8 -*-
import os
import random
import queue
import warnings
from enum import Enum

from hok.gamecore_utils import (
    GameLauncher,
    GameLauncherRemote,
)


class ResponceType(Enum):
    NONE = 0
    DEFAULT = 1
    CACHED = 2
    GAMEOVER = 3


import numpy as np
import hok.lib.interface as interface


class HoK1v1:
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
    OBS_SHAPE = [453]
    PLAYER_NUM = 2
    REPEAT_INIT = 30
    HERO_ID_INDEX_DICT = {
        112: 0,
        121: 1,
        123: 2,
        131: 3,
        132: 4,
        133: 5,
        140: 6,
        141: 7,
        146: 8,
        150: 9,
        154: 10,
        157: 11,
        163: 12,
        169: 13,
        175: 14,
        182: 15,
        193: 16,
        199: 17,
        502: 18,
        513: 19,
    }

    @staticmethod
    def load_game(**kwargs):
        return HoK1v1(**kwargs)

    def __init__(
        self,
        runtime_id,
        game_log_path,
        gamecore_path,
        config_path=None,
        eval_mode=False,
        predict_frequency=3,
        render=None,
        remote_param=None,
    ):
        """
        Init the environment

        :param runtime_id:
        :param game_log_path: game log path
        :param gamecore_path:  gamecore installation path
        :param config_path: configuration path
        :param eval_mode: whether evaluation mode
        :param predict_frequency: predict frequency
        :param render: renderer
        :param remote_mode: 0: local gamecore, 1:

        """

        if config_path is None:
            config_path = "config.dat"
        config_path = os.path.join(os.path.dirname(__file__), config_path)

        self.runtime_id = runtime_id
        self.player_ids = {}
        self.camp_list = []
        self.player_list = [None] * self.PLAYER_NUM
        self.player_pos = [
            None,
        ] * self.PLAYER_NUM

        self.lib_processor = interface.Interface()
        print("load config.dat: ", __file__)
        self.lib_processor.Init(config_path)
        self.is_gameover = True

        self.action_size = self.LABEL_SIZE_LIST

        # init list for split legal action
        self.legal_action_split_size = []
        tmp = 0
        for _, s in enumerate(self.action_size):
            tmp += s
            self.legal_action_split_size.append(tmp)
        self.legal_action_split_size = self.legal_action_split_size

        self.player_masks = {}

        self.cur_frame_no = -1

        self.cur_state = None
        self.cur_req_pb = None
        self.cur_sgame_ids = [None] * self.PLAYER_NUM
        self.eval_mode = eval_mode

        self.repeat_init = [0] * self.PLAYER_NUM

        self.predict_frequency = predict_frequency
        self.request_latency = 1
        self.render = render

        self._act_que = []
        self.start_frame = -1

        if remote_param is None or remote_param["remote_mode"] == 0:
            self.game_launcher = GameLauncher(
                runtime_id,
                game_log_path,
                gamecore_path,
                lib_processor=self.lib_processor,
            )
        else:
            self.game_launcher = GameLauncherRemote(
                runtime_id,
                game_log_path,
                gamecore_path,
                local_server=(remote_param["remote_mode"] == 1),
                launch_server=remote_param["gc_server_addr"],
                aiserver_ip=remote_param["ai_server_addr"],
                lib_processor=self.lib_processor,
            )
            # print(runtime_id, game_log_path, gamecore_path)

        self.button_names = [
            "None1",
            "None2",
            "Move",
            "Attack",
            "Skill1",
            "Skill2",
            "Skill3",
            "HealSkill",
            "ChosenSkill",
            "Recall",
            "Skill4",
            "EquipSkill",
        ]
        self.category_names = [
            "which_button",
            "move_x",
            "move_z",
            "kill_x",
            "kill_z",
            "target",
        ]
        self.wait_game_max_timeout = 30

    # util_functions
    def get_random_action(self, info):
        """
        Get random and legal actions based on current states
        """
        actions = []

        shapes = self.action_space()

        split_array = shapes.copy()[:-1]
        for i in range(1, len(split_array)):
            split_array[i] = split_array[i - 1] + split_array[i]

        sub_actions = self.get_subsequent_actions(info)
        for id in range(self.PLAYER_NUM):
            legal_action = np.split(info[id]["legal_action"], split_array)
            legal_action[-1] = legal_action[-1].reshape(
                self.LABEL_SIZE_LIST[0], self.LABEL_SIZE_LIST[-1]
            )

            action = [0] * len(self.LABEL_SIZE_LIST)
            button = random.choice(sub_actions[id]["which_button"])
            action[0] = button
            for category_0 in range(len(self.category_names)):
                if (
                    self.category_names[category_0]
                    == sub_actions[id][self.button_names[button]].keys()
                ):
                    action[category_0] = random.choice(
                        sub_actions[id][self.button_names[button]][category_0]
                    )
                else:
                    if category_0 != len(self.LABEL_SIZE_LIST) - 1:
                        action[category_0] = random.choice(
                            [
                                index
                                for index in range(len(legal_action[category_0]))
                                if legal_action[category_0][index] == 1
                            ]
                        )
                    else:
                        action[category_0] = random.choice(
                            [
                                index
                                for index in range(
                                    len(legal_action[category_0][button])
                                )
                                if legal_action[category_0][button][index] == 1
                            ]
                        )
            actions.append(tuple(action))
        return actions

    def get_noop_action(self, obs=None, info=None):
        """
        get non-op action
        """
        actions = []
        for _ in range(self.PLAYER_NUM):
            actions.append([0] * len(self.action_space()))
        return actions

    def get_subsequent_actions(self, info):
        """
        get the legal actions, first key is the button, second key is the botton, how and who
        """
        shapes = self.action_space()

        split_array = shapes.copy()[:-1]
        for i in range(1, len(split_array)):
            split_array[i] = split_array[i - 1] + split_array[i]

        subsequent_actions = [None] * self.PLAYER_NUM
        for id in range(self.PLAYER_NUM):
            if not self.need_predict(id):
                continue
            subsequent_actions[id] = dict()
            # first, make keys as the which buttion
            sub_action = info[id]["sub_action_mask"]

            legal_action = np.split(info[id]["legal_action"], split_array)
            legal_action[-1] = legal_action[-1].reshape(
                self.LABEL_SIZE_LIST[0], self.LABEL_SIZE_LIST[-1]
            )

            allowed_button = [
                index
                for index in range(len(legal_action[0]))
                if legal_action[0][index] == 1
            ]
            subsequent_actions[id][self.category_names[0]] = allowed_button
            subsequent_actions[id]["legal_action"] = legal_action
            for button in allowed_button:
                button_name = self.button_names[button]
                allowed_categories = [
                    index
                    for index in range(len(sub_action[button]))
                    if sub_action[button][index] == 1
                ]
                subsequent_actions[id][button_name] = dict()
                for category in allowed_categories:
                    category_name = self.category_names[category]
                    if category == 0:
                        continue
                    elif category == len(self.LABEL_SIZE_LIST) - 1:
                        subsequent_actions[id][button_name][category_name] = [
                            index
                            for index in range(len(legal_action[category][button]))
                            if legal_action[category][button][index] == 1
                        ]
                    else:
                        subsequent_actions[id][button_name][category_name] = [
                            index
                            for index in range(len(legal_action[category]))
                            if legal_action[category][index] == 1
                        ]

        return subsequent_actions

    def action_space(self):
        """
        the action space of an agent in the environment
        """
        return self.LABEL_SIZE_LIST

    def obs_space(self):
        """
        the observation space of the agent in the environment
        """
        return self.OBS_SHAPE

    def _state2ret(self, states):
        obs = [None] * self.PLAYER_NUM
        reward = [0.0] * self.PLAYER_NUM
        done = [False] * self.PLAYER_NUM
        info = states
        for i in range(self.PLAYER_NUM):
            if not self.need_predict(i):
                continue
            obs[i] = states[i]["observation"]
            temp_reward = states[i]["reward"][:6] + states[i]["reward"][7:]
            reward[i] = temp_reward
            done[i] = states[i]["done"]

        return obs, reward, done, info

    def _split_legal_action(self, la, button):
        tmp = np.split(la, self.legal_action_split_size[:-1])
        tmp[-1] = tmp[-1].reshape(-1, self.LABEL_SIZE_LIST[-1])[button]
        return tmp

    def _check_action(self, actions):
        # check whether the actions are legal
        for i, act in enumerate(actions):
            if not self.need_predict(i):
                continue
            legal = self.cur_state[i]["legal_action"]
            legal = self._split_legal_action(legal, act[0])
            sub = self.cur_state[i]["sub_action_mask"][int(act[0])]

            for j in range(6):
                a = int(act[j])
                if (a < 0 or a >= len(legal[j])) or legal[j][a] == 0:
                    warnings.warn(
                        "Agent[{}] is passed with an illegal action {} No.{}:[{}], legal: {}, all: {}, sub: {}".format(
                            i, self.category_names[j], j, a, legal[j], act, sub
                        )
                    )

    def step(self, actions):
        """
        pass actions to environment
        """
        self._check_action(actions)
        self._step_action(actions)
        state, req_pb = self._step_feature()
        self._update_gameover(state, req_pb)
        # return state, req_pb
        self.cur_req_pb = req_pb
        self.cur_state = state
        return self._state2ret(state)

    def _state_tuple2np(self, states):
        states = list(states)
        for _, state in enumerate(states):
            if state is None:
                continue
            for k in state:
                if isinstance(state[k], tuple) and k in ["legal_action"]:
                    state[k] = np.array(state[k])
                if isinstance(state[k], tuple) and k in ["observation"]:
                    state[k] = np.array(state[k])
                    hero_id = self.game_launcher.config_modifier.hero_info[0]["hero"]
                    hero_id_vec = np.zeros(
                        [
                            len(self.HERO_ID_INDEX_DICT),
                        ],
                        dtype=np.float,
                    )
                    hero_id_vec[self.HERO_ID_INDEX_DICT[hero_id]] = 1
                    # print("-------hero_id-------",hero_id_vec)

                    state[k] = np.concatenate((state[k], hero_id_vec), axis=0)
                if isinstance(state[k], dict) and k in ["sub_action_mask"]:
                    # tmp_tuple = {}
                    for i in state[k]:
                        state[k][i] = np.array(state[k][i])

        return states

    def _update_gameover(self, state, req_pb):
        for i in range(self.PLAYER_NUM):
            if not self.need_predict(i):
                continue
            state[i]["done"] = state[i]["done"] or req_pb[i].gameover

    def _format_actions(self, actions):
        # check whether the actions are within defined range, and format into gamecore actions
        rp_actions = []
        for i, action in enumerate(actions):
            if self.player_masks[i]:
                action = [0.0] * 6
            # formulation check
            if isinstance(action, (tuple, list)):
                if not len(action) == 6:
                    assert False, "action[{}] length incorrect: {}, but expect 6."
                action = np.array(action)
            elif isinstance(action, np.ndarray):
                if not (len(action.shape) == 1 and action.shape[0] == 6):
                    assert (
                        False
                    ), "action[{}] shape incorrect: {}, but expect [6].".format(
                        i, action.shape
                    )
            else:
                assert False, "invalid action[{}] type of {}".format(i, type(action))
            old_action = action
            action = []
            for j, act in enumerate(old_action):
                assert (
                    0 <= act < self.action_size[j]
                ), "Action[{}] {}: {} not in [0,{})".format(
                    i, j, act, self.action_size[j]
                )
                action.append((act,))
            action = tuple(action)
            rp_act = (
                (
                    (0,) * 12,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 8,
                )
                + action
                + ((0,),)
            )
            rp_actions.append(rp_act)
        return tuple(rp_actions)

    def _step_action(self, actions):
        # 6 actions: top, move_x, move_z, skill_x, skill_z, target
        # msg format to RP:
        #   logp: 0(11) 1(15) 2(15) 3(15) 4(15), target(7)
        #   sample: 0, 1, 2, 3, 4, target
        #   value: [1]
        rp_actions = self._format_actions(actions)
        s_msgs = [None] * self.PLAYER_NUM
        for id in range(self.PLAYER_NUM):
            if not self.need_predict(id):
                # silently skip if is common ai
                continue

            ret_code, resp_id = self.lib_processor.ResultProcess(
                rp_actions[id : id + 1], self.cur_sgame_ids[id]
            )

            assert (
                ret_code == interface.PROCESS_ACTION_SUCCESS
            ), "process action failed: {}".format(ret_code)

            s_msgs[id] = (ResponceType.CACHED, resp_id)
        self._send_all(s_msgs)
        # self._send(ret[1], id)

    def need_predict(self, id):
        """
        Whether the player needs passed action
        """
        return not self.player_masks[id]

    def _init_hero_info(self, pb, id):
        if self.player_list[id] is not None:
            return
        if 0 < len(pb.command_info_list):
            # set current player
            for cmd in pb.command_info_list:
                print("cmd.player_id", cmd.player_id)
                self.player_list[id] = cmd.player_id

            # get camp info
            for hero_state in pb.hero_list:
                if hero_state.runtime_id == self.player_list[id]:
                    self.player_ids.setdefault(hero_state.camp, []).append(
                        hero_state.runtime_id
                    )
                    self.camp_list.append(hero_state.camp)
                    self.player_camp[hero_state.runtime_id] = hero_state.camp

    def _recv_all(self):
        pbs = []
        parse_ret = []
        for id in range(self.PLAYER_NUM):
            if not self.need_predict(id):
                pbs.append(None)
                parse_ret.append(None)
                continue
            # import pdb;pdb.set_trace();
            parse_state, sgame_id = self.lib_processor.RecvAIFrameState(
                self.game_launcher.addrs[id]
            )
            self.cur_sgame_ids[id] = sgame_id

            assert (
                parse_state == interface.PARSE_CONTINUE
                or parse_state == interface.PARSE_NONE_ACTION
            ), ("step failed: %s" % parse_state)

            req_pb = None
            if parse_state == interface.PARSE_CONTINUE:
                req_pb = self.lib_processor.GetAIFrameState(sgame_id)
                assert req_pb is not None, "GetAIFrameState failed"

            if self._act_que[id].empty():
                # first frame!
                # update player info with info
                self._init_hero_info(req_pb, id)
                for _ in range(self.request_latency):
                    self._act_que[id].put((ResponceType.NONE, -1))

            smsg = self._act_que[id].get()

            self._send(smsg, id)
            pbs.append(req_pb)
            parse_ret.append((parse_state, sgame_id))
        return parse_ret, pbs

    def _send_all(self, msgs, ignore_none=False):
        # msgs: list of (ResponceType, param)
        # only put rsp_msg in buffer
        for id in range(self.PLAYER_NUM):
            if not self.need_predict(id):
                continue
            if msgs[id] is None:
                if ignore_none:
                    # self._send(self._get_rsp_pb(id).SerializeToString(), id)
                    self._act_que[id].put((ResponceType.NONE, -1))
                else:
                    assert False, "send None at {}".format(id)
            else:
                self._act_que[id].put(msgs[id])

    def _step_feature(self, first_frame=False):
        while True:
            ret_flag = [False] * self.PLAYER_NUM
            ret_num = 0
            real_num = 0
            states = [None] * self.PLAYER_NUM

            r_msgs, req_pbs = self._recv_all()
            # print("_step_feature: ", req_pbs[0].frame_no)
            s_msgs = [None] * self.PLAYER_NUM

            for id in range(self.PLAYER_NUM):
                if not self.need_predict(id):
                    # silently skip if is common ai
                    continue
                real_num += 1
                req_pb = req_pbs[id]
                self.cur_frame_no = req_pb.frame_no
                if first_frame:
                    self.start_frame = self.cur_frame_no

                if req_pb.gameover:
                    print("game is over!")
                    self.is_gameover = True

                if (not first_frame) and (
                    self.cur_frame_no - self.start_frame
                ) % self.predict_frequency > 0:
                    # skip this non-predict frame
                    # if gameover, skip c++ process, so checking gameover with py code is necessary.
                    if not self.is_gameover:
                        s_msgs[id] = (ResponceType.NONE, -1)
                        continue

                parse_state, sgame_id = r_msgs[id]
                ret = self.lib_processor.FeatureProcess(parse_state, sgame_id)
                # Failed, return no action
                if ret[0] == 0:
                    # LOG.error("step failed: {}".format(ret[1]))
                    assert False, "step failed: {}".format(ret[1])
                if ret[0] == 1:
                    # directly receive msg again!
                    assert False, "Parsing gameover information, receive msg again!"
                    # LOG.error("Parsing gameover information, receive msg again!")
                elif ret[0] == 2:
                    # SEND_CCD_ONE_HERO, get normal feature vector, break
                    state = self._state_tuple2np(ret[1:])[0]
                    if req_pb.gameover:
                        print(
                            "gameover at frameno {} of {}!".format(req_pb.frame_no, id)
                        )
                    # LOG.debug("Parsing normal feature of len({}).".format(len(states)))
                    ret_num += 1
                    state["req_pb"] = req_pb
                    state["sgame_id"] = sgame_id
                    states[id] = state
                    ret_flag[id] = True
                    s_msgs[id] = None
                    # return states, req_pb
                elif ret[0] == 3 or ret[0] == 4 or ret[0] == 5:
                    # SEND_CCD_FIVE_HERO
                    self.repeat_init[id] = 0
                    if ret[0] == 3:
                        s_msgs[id] = (ResponceType.DEFAULT, -1)
                    elif ret[0] == 4:
                        s_msgs[id] = (ResponceType.NONE, -1)
                    elif ret[0] == 5:
                        s_msgs[id] = (ResponceType.CACHED, int(ret[1]))

            if self.render is not None:
                # print("render frame!")
                self.render.draw_frame(req_pbs, self.cur_frame_no)

            if first_frame:
                if ret_num < real_num:
                    self._send_all(s_msgs, ignore_none=True)
                    continue
            elif (self.cur_frame_no - self.start_frame) % self.predict_frequency > 0:
                if not self.is_gameover:
                    self._send_all(s_msgs)
                    continue

            return states, req_pbs

    def close_game(self):
        """
        close game by sending signals to gamecore
        """
        while not self.is_gameover:
            print("game not end, send close game at first", self.cur_frame_no)
            for i in range(self.PLAYER_NUM):

                # silently skip if is common ai
                if not self.need_predict(i):
                    continue

                parse_state, sgame_id = self.lib_processor.RecvAIFrameState(
                    self.game_launcher.addrs[i]
                )
                if parse_state != interface.PARSE_CONTINUE:
                    continue
                req_pb = self.lib_processor.GetAIFrameState(sgame_id)
                if req_pb.gameover:
                    self.is_gameover = True

                self._gameover(i, True)

        # wait game over
        self.game_launcher.wait_game(self.wait_game_max_timeout)

        # force close
        self.game_launcher.close_game(keep_zmq=True)

    def reset(
        self,
        config_dicts=None,
        common_config=None,
        use_common_ai=None,
        eval=None,
        render=None,
        game_id=None,
    ):
        """
        resetting the environment

        :param config_dicts: configure hero and chosen skill
        :param common_config: confiture common configs like the request frequency
        :param use_common_ai: whether to use common_ai
        :param eval: whether is evaluation mode
        :param render: whether use renderer
        :param game_id: game id
        :return: obs, reward, done, info
        """
        if use_common_ai is None:
            use_common_ai = [False] * self.PLAYER_NUM
        if common_config is None:
            common_config = {}
        if config_dicts is None:
            config_dicts = [{} for _ in range(self.PLAYER_NUM)]
        if eval is not None:
            self.eval_mode = eval
        # reset infos
        sgame_ids = []
        if self.cur_sgame_ids:
            for sgame_id in self.cur_sgame_ids:
                if sgame_id:
                    sgame_ids.append(sgame_id)
        self.lib_processor.Reset(self.eval_mode, sgame_ids)

        self.last_predict_frame = 0
        self.player_ids = {}
        self.player_camp = {}
        self.camp_list = []
        self.reward = []
        self.player_list = [None] * self.PLAYER_NUM
        self.repeat_init = [0] * self.PLAYER_NUM
        # print("Reset Done, game_launcher {} start".format(game_id))
        if game_id is None:
            game_id = self.game_launcher.generate_game_id()
        self.game_id = game_id
        common_config["game_id"] = self.game_id
        self.is_gameover = False

        self.player_masks = use_common_ai.copy()

        # update gamecore config file and start it
        for i in range(self.PLAYER_NUM):
            config_dicts[i]["use_common_ai"] = use_common_ai[i]
            if config_dicts[i].get("port") is None:
                config_dicts[i]["port"] = 35300 + self.PLAYER_NUM * self.runtime_id + i

        if self.render is not None:
            common_config["request_freq"] = 1
            self.request_latency = 1
            # render process may cause timeout, set it to 5 sec.
            common_config["timeout"] = 5000
        else:
            common_config["request_freq"] = 1
            self.request_latency = 1

        # print("game_launcher with config_dicts: {},\n common_config: {}".format(config_dicts, common_config))
        self.game_launcher.start(config_dicts, common_config)
        # print("game_launcher done")
        self._act_que = [queue.Queue() for _ in range(self.PLAYER_NUM)]

        states, req_pbs = self._step_feature(True)
        self._update_gameover(states, req_pbs)

        # return state, req_pb
        self.cur_state = states
        self.cur_req_pb = req_pbs
        return self._state2ret(states)

    def _send(self, msg, id):
        send_type, msg_id = msg
        ret = None
        if send_type == ResponceType.GAMEOVER:
            ret = self.lib_processor.SendGameoverResp(
                self.game_launcher.addrs[id], self.cur_sgame_ids[id]
            )
        elif send_type == ResponceType.DEFAULT:
            ret = self.lib_processor.SendDefaultResp(self.game_launcher.addrs[id])
        elif send_type == ResponceType.NONE:
            ret = self.lib_processor.SendNoneResp(
                self.game_launcher.addrs[id], self.cur_sgame_ids[id]
            )
        elif send_type == ResponceType.CACHED:
            ret = self.lib_processor.SendResp(
                self.game_launcher.addrs[id], self.cur_sgame_ids[id], msg_id
            )
        else:
            assert False, "Unknown ResponceType: %s" % send_type

        assert ret == interface.SEND_SUCCESS, "Send resp failed: %s" % ret

    def _get_move_pb(self, id, d):
        response = AIServerResponse()
        player_id = self.player_list[id]
        # for testing
        cmd_list = AICommandInfo()
        cmd_list.player_id = player_id
        cmd_list.cmd_info.command_type = COMMAND_TYPE_MoveDir
        cmd_list.cmd_info.move_dir.degree = d
        response.cmd_list.extend([cmd_list])
        return response

    def _gameover(self, id, force_send=False):
        self._send((ResponceType.GAMEOVER, -1), id)
