import os
import json

from enum import Enum


class Action(Enum):
    WHICH_BUTTON = 0
    MOVE = 1
    OFFSET_X = 2
    OFFSET_Z = 3
    TARGET = 4


class Button(Enum):
    NONE_ACTION = 0
    NONE_ACTION_2 = 1
    MOVE = 2
    NORMAL_ATTACK = 3
    SKILL_1 = 4
    SKILL_2 = 5
    SKILL_3 = 6
    SKILL_4 = 7
    CHOSEN_SKILL = 8
    RECALLING_TO_THE_BASE = 9
    EQUIPMENT_SKILL = 10
    HEAL_SKILL = 11
    FRIEND_SKILL = 12


_dir = [None] + [(180 - 15 * i) % 360 for i in range(24)]


class Direction(Enum):
    DIR_0 = 0
    DIR_1 = 1
    DIR_2 = 2
    DIR_3 = 3
    DIR_4 = 4
    DIR_5 = 5
    DIR_6 = 6
    DIR_7 = 7
    DIR_8 = 8
    DIR_9 = 9
    DIR_10 = 10
    DIR_11 = 11
    DIR_12 = 12
    DIR_13 = 13
    DIR_14 = 14
    DIR_15 = 15
    DIR_16 = 16
    DIR_17 = 17
    DIR_18 = 18
    DIR_19 = 19
    DIR_20 = 20
    DIR_21 = 21
    DIR_22 = 22
    DIR_23 = 23
    DIR_24 = 24

    def to_dir(self):
        return _dir[self.value]


class TargetType(Enum):
    NONE = 0
    ENEMY_HERO = 1
    FRIEND_HERO = 2
    SELF = 3
    MONSTER = 4
    ENEMY_MINIONS = 5
    ENEMY_TURRET = 6
    UNKNOWN = 7


class Target(Enum):
    NONE = 0
    Enemy_Hero_0 = 1
    Enemy_Hero_1 = 2
    Enemy_Hero_2 = 3
    Friend_Hero_0 = 4
    Friend_Hero_1 = 5
    Friend_Hero_2 = 6
    SELF = 7
    Red_Statue = 8
    Red_Statue_Enemy = 9
    Greater_Demon_Pioneer = 10
    Greater_Demon_Pioneer_Enemy = 11
    Lesser_Demon_Pioneer = 12
    Lesser_Demon_Pioneer_Enemy = 13
    Greater_Demon_Archer = 14
    Greater_Demon_Archer_Enemy = 15
    Lesser_Demon_Archer = 16
    Lesser_Demon_Archer_Enemy = 17
    Greater_Whitetail_Deer = 18
    Greater_Whitetail_Deer_Enemy = 19
    Lesser_Whitetail_Deer = 20
    Lesser_Whitetail_Deer_Enemy = 21
    Greater_Shadow_Wolf = 22
    Greater_Shadow_Wolf_Enemy = 23
    Lesser_Shadow_Wolf = 24
    Lesser_Shadow_Wolf_Enemy = 25
    Treasure_Thief = 26
    Tyrant = 27
    ENEMY_MINION_0 = 28
    ENEMY_MINION_1 = 29
    ENEMY_MINION_2 = 30
    ENEMY_MINION_3 = 31
    ENEMY_MINION_4 = 32
    ENEMY_MINION_5 = 33
    ENEMY_MINION_6 = 34
    ENEMY_MINION_7 = 35
    ENEMY_MINION_8 = 36
    ENEMY_MINION_9 = 37
    ENEMY_TURRET = 38

    def get_target_type(self):
        if self.value == 0:
            return TargetType.NONE
        elif self.value <= 3:
            return TargetType.ENEMY_HERO
        elif self.value <= 6:
            return TargetType.FRIEND_HERO
        elif self.value == 7:
            return TargetType.SELF
        elif self.value <= 27:
            return TargetType.MONSTER
        elif self.value <= 37:
            return TargetType.ENEMY_MINIONS
        elif self.value == 38:
            return TargetType.ENEMY_TURRET
        else:
            return TargetType.UNKNOWN

    def get_config_id(
        self, ego_hero_config_ids, enemy_hero_config_ids, self_hero_config_id
    ):
        ego_hero_config_ids.sort()
        enemy_hero_config_ids.sort()
        ego_hero_config_ids += [-1] * (3 - len(ego_hero_config_ids))
        enemy_hero_config_ids += [-1] * (3 - len(enemy_hero_config_ids))

        target_to_config_id = {
            Target.NONE: -1,
            Target.Enemy_Hero_0: -1,
            Target.Enemy_Hero_1: -1,
            Target.Enemy_Hero_2: -1,
            Target.Friend_Hero_0: -1,
            Target.Friend_Hero_1: -1,
            Target.Friend_Hero_2: -1,
            Target.SELF: -1,
            Target.Red_Statue: 49,
            Target.Red_Statue_Enemy: 49,
            Target.Greater_Demon_Pioneer: 30,
            Target.Greater_Demon_Pioneer_Enemy: 30,
            Target.Lesser_Demon_Pioneer: 31,
            Target.Lesser_Demon_Pioneer_Enemy: 31,
            Target.Greater_Demon_Archer: 32,
            Target.Greater_Demon_Archer_Enemy: 32,
            Target.Lesser_Demon_Archer: 33,
            Target.Lesser_Demon_Archer_Enemy: 33,
            Target.Greater_Whitetail_Deer: 42,
            Target.Greater_Whitetail_Deer_Enemy: 42,
            Target.Lesser_Whitetail_Deer: 43,
            Target.Lesser_Whitetail_Deer_Enemy: 43,
            Target.Greater_Shadow_Wolf: 44,
            Target.Greater_Shadow_Wolf_Enemy: 44,
            Target.Lesser_Shadow_Wolf: 45,
            Target.Lesser_Shadow_Wolf_Enemy: 45,
            Target.Treasure_Thief: 59,
            Target.Tyrant: 122,
        }
        hero_configs = (
            [-1] + enemy_hero_config_ids + ego_hero_config_ids + [self_hero_config_id]
        )
        for target, config_id in enumerate(hero_configs):
            target_to_config_id[Target(target)] = config_id

        return target_to_config_id.get(self, -1)


class DumpProbs:
    def __init__(self, req_pb, features, process_result):
        self.req_pb = req_pb
        self.features = features
        self.process_result = process_result

    def save_to_file(self, output_file):
        data = self.parse_prob()

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "a") as f:
            json.dump(data, f)
            f.write("\n")

    def _parse_button(self, button, hero_idx):
        button = Button(button)
        return {"name": button.name, "value": button.value}

    def _parse_move(self, move, hero_idx):
        move = Direction(move)
        return {"name": move.name, "value": move.value, "direction": move.to_dir()}

    def _get_hero_config_ids(self, hero_idx):
        self_hero_config_id = -1
        ego_camp_id = self.features[hero_idx].camp_id
        ego_hero_config_ids = []
        enemy_hero_config_ids = []

        for hero in self.req_pb.hero_list:
            if hero.camp == ego_camp_id:
                ego_hero_config_ids.append(hero.config_id)
            else:
                enemy_hero_config_ids.append(hero.config_id)

            if hero.runtime_id == self.features[hero_idx].hero_runtime_id:
                self_hero_config_id = hero.config_id
        ego_hero_config_ids.sort()
        enemy_hero_config_ids.sort()
        return ego_hero_config_ids, enemy_hero_config_ids, self_hero_config_id

    def _parse_target(self, target, hero_idx):
        target = Target(target)

        return {
            "name": target.name,
            "value": target.value,
            "type": target.get_target_type().name,
            "config_id": target.get_config_id(*self._get_hero_config_ids(hero_idx)),
        }

    def get_action_parse_fn(self, action):
        def _same(x, hero_idx):
            return {"name": "{}_{}".format(action.name, x), "value": x}

        ret = {
            Action.WHICH_BUTTON: self._parse_button,
            Action.MOVE: self._parse_move,
            Action.TARGET: self._parse_target,
        }
        return ret.get(Action(action), _same)

    def _parse_legal_action(self, values, hero_idx, action_parser):
        ret = {}
        for i, x in enumerate(values):
            name = action_parser(i, hero_idx)["name"]
            ret[name] = x == 1
        return ret

    def get_legal_action_pasre_fn(self, action_parser):
        def _parse_legal_action(values, hero_idx):
            return self._parse_legal_action(values, hero_idx, action_parser)

        return _parse_legal_action

    def _parse_prob(self, values, hero_idx, action_parser):
        top_3 = sorted([(x, i) for i, x in enumerate(values)], reverse=True)[:3]

        ret_top = [{"prob": prob, **action_parser(i, hero_idx)} for prob, i in top_3]

        return ret_top

    def get_probs_pasre_fn(self, action, action_parser):
        def _parse_prob(values, hero_idx):
            return self._parse_prob(values, hero_idx, action_parser)

        return _parse_prob

    def parse_prob(self):
        runtime_to_config = {
            hero_state.runtime_id: hero_state.config_id
            for hero_state in self.req_pb.hero_list
        }

        heros = []
        for hero_idx in range(len(self.process_result)):
            config_id = runtime_to_config[
                self.features[hero_idx].hero_runtime_id
            ]

            _final_prob_list = self.process_result[hero_idx].final_prob_list
            # [len(x) for x in final_prob_list] == [13, 25, 42, 42, 39, 1]

            _actions = self.process_result[hero_idx].actions
            _legal_action = self.process_result[hero_idx].legal_action
            _sub_actions = self.process_result[hero_idx].sub_actions
            # len(sub_actions) == len(_legal_action) == len(_actions) == 5
            # [len(x) for x in _legal_action] == [13, 25, 42, 42, 39]

            actions, sub_actions, legal_action, probs = {}, {}, {}, {}
            # parse actions
            for action_type in Action:
                # _actions => action:
                #    (2, 7, 11, 31, 0) =>
                #    {
                #        "WHICH_BUTTON": {"name": "MOVE", "value": 2},
                #        "MOVE": {"name": "DIR_7", "value": 7, "direction": 75},
                #        "OFFSET_X": 11,
                #        "OFFSET_Z": 31,
                #        "TARGET": {"name": "NONE", "value": 0, "type": "None", "config_id": -1},
                #    }
                action_value = _actions[action_type.value]
                action_value_parser = self.get_action_parse_fn(action_type)
                actions[action_type.name] = action_value_parser(action_value, hero_idx)

                # _subaction => sub_actions:
                #    (1, 1, 0, 0 ,0) =>
                #    {
                #        "WHICH_BUTTON": True,
                #        "MOVE": True,
                #        "OFFSET_X": False,
                #        "OFFSET_Z": False,
                #        "TARGET": False,
                #    }
                sub_actions[action_type.name] = _sub_actions[action_type.value] == 1

                # _legal_action
                legal_action_parser = self.get_legal_action_pasre_fn(
                    action_value_parser
                )
                legal_action[action_type.name] = legal_action_parser(
                    _legal_action[action_type.value], hero_idx
                )

                # _final_prob_list
                probs_parser = self.get_probs_pasre_fn(action_type, action_value_parser)
                probs[action_type.name] = probs_parser(
                    _final_prob_list[action_type.value], hero_idx
                )

            hero = {
                "config_id": config_id,
                "sub_actions": sub_actions,
                "legal_action": legal_action,
                "actions": actions,
                "probs": probs,
                # "probs": {
                #     "which_button": final_prob_list[0],
                #     "move": final_prob_list[1],
                #     "offset_x": final_prob_list[2],
                #     "offset_z": final_prob_list[3],
                #     "target": final_prob_list[4],
                # },
            }
            heros.append(hero)

        data = {
            "sgame_id": self.req_pb.sgame_id,
            "frame_no": self.req_pb.frame_no,
            "camp_id": self.features[0].camp_id,
            "heros": heros,
        }
        return data


if __name__ == "__main__":
    from action_space_test import (
        req_pb_test_data,
        features_test_data,
        process_result_test_data,
    )

    dump_probs = DumpProbs(
        req_pb_test_data, features_test_data, process_result_test_data
    )
    dump_probs.save_to_file("probs.json")
