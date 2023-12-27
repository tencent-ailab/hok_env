# use default agent:
# from hok.hok3v3.agent import Agent as Agent

import math

from numpy.random import rand
import numpy as np

# custom agent
from agent.agent import Agent as BaseAgent


pred_ret_shape = [(1, 162)] * 3
lstm_cell_shape = [(1, 16), (1, 16)]

tower_locations = [48020, 6020]

# Use log to print information on the terminal
from rl_framework.common.logging import logger as LOG

from hok.hok3v3.lib.lib3v3 import (
    PLAYERCAMP_1,
    PLAYERCAMP_2,
    PLAYERCAMP_MID,
    ACTOR_SOLDIER,
    ACTOR_TOWER,
    ACTOR_TOWER_HIGH,
    ACTOR_TOWER_SPRING,
    ACTOR_CRYSTAL,
)


def get_monster_camp(monster):
    if monster.config_id in [59, 122]:
        return PLAYERCAMP_MID
    return PLAYERCAMP_1 if (monster.location.x) < 0 else PLAYERCAMP_2


def get_monster_offset_dict(ego_camp_id, enemy_camp_id):
    # Save different types of monster and their corresponding index in target dimension in offset_dict.
    return {
        49: {
            ego_camp_id: 0,
            enemy_camp_id: 1,
        },
        30: {
            ego_camp_id: 2,
            enemy_camp_id: 3,
        },
        31: {
            ego_camp_id: 4,
            enemy_camp_id: 5,
        },
        32: {
            ego_camp_id: 6,
            enemy_camp_id: 7,
        },
        33: {
            ego_camp_id: 8,
            enemy_camp_id: 9,
        },
        42: {
            ego_camp_id: 10,
            enemy_camp_id: 11,
        },
        43: {
            ego_camp_id: 12,
            enemy_camp_id: 13,
        },
        44: {
            ego_camp_id: 14,
            enemy_camp_id: 15,
        },
        45: {
            ego_camp_id: 16,
            enemy_camp_id: 17,
        },
        59: {
            PLAYERCAMP_MID: 18,
        },
        122: {
            PLAYERCAMP_MID: 19,
        },
    }


class Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        kwargs["rule_only"] = True
        super().__init__(*args, **kwargs)

        self.camp = None
        self.distance_threshold = 6000
        self.ego_camp_id = PLAYERCAMP_1
        self.enemy_camp_id = PLAYERCAMP_2

        self.organ_type = [
            ACTOR_TOWER,
            ACTOR_TOWER_HIGH,
            ACTOR_TOWER_SPRING,
            ACTOR_CRYSTAL,
        ]
        self.soldier_type = [ACTOR_SOLDIER]
        self.ego_camp = []
        self.enemy_camp = []
        self.enemy_soldier = []
        self.monster = []
        self.enemy_organ = []

        # A list which is used to keep the counterpart index of monster in target dimension
        self.monster_idx_offset = []

    def _predict_process_torch(self, features, frame_state, runtime_ids):
        return self._predict_process(features, frame_state, runtime_ids)

    def _predict_process(self, features, frame_state, runtime_ids):
        # predict using model
        # pred_ret, lstm_info = super()._predict_process(
        #    hero_data_list, frame_state, runtime_ids
        # )

        # Randomly initialize pred_ret
        pred_ret = []
        for shape in pred_ret_shape:
            pred_ret.append(rand(*shape).astype("float32"))
        lstm_info = []
        for shape in lstm_cell_shape:
            lstm_info.append(np.zeros(shape).astype("float32"))

        self._generate_rule_actions(frame_state, pred_ret, runtime_ids)

        return pred_ret, lstm_info

    def _generate_rule_actions(self, frame_state, pred_ret, runtime_ids):
        # Get index of heroes, monsters, soldiers and organs in frame_state
        self.ego_camp, self.enemy_camp = self.get_ego_enemy(frame_state, runtime_ids)
        (
            self.enemy_soldier,
            self.monster,
            self.monster_idx_offset,
            self.enemy_organ,
        ) = self.get_npc(frame_state)

        # Get location of heroes, monsters, soldiers and organs
        ids, location = self.get_location(frame_state)

        for i, hero_idx in enumerate(self.ego_camp):
            ego_loc = frame_state.hero_list[hero_idx].location
            ego_loc_x = ego_loc.x
            ego_loc_z = ego_loc.z
            distances = []

            # Calculate distance from hero to enemy heroes, enemy soldiers, monsters and enemy organs
            for loc in location:
                x_diff = loc[0] - ego_loc_x
                z_diff = loc[1] - ego_loc_z
                dis_diff_hero2target = np.sqrt(x_diff**2 + z_diff**2)

                distances.append(dis_diff_hero2target)

            # Try to define if there is a target whose distance from the hero is not 0
            flags = np.array(distances) != 0

            if any(flags):
                # Get the distance and index of target who is nearest to hero
                min_dis = min(np.array(distances)[flags])
                target_idx = distances.index(min_dis)
                # If the minimal distance is less than the distance threshold, then launch an attack
                if min_dis < self.distance_threshold:
                    self.release_skill(pred_ret, target_idx, ids, index=i)
                else:
                    # if the distance is too far, try to move towards enemy's organ
                    if self.ego_camp_id == PLAYERCAMP_2:
                        # We try to convert the red team's frame state data for symmetric processing so that we can calculate in the same way.
                        x_diff = tower_locations[0] - (-ego_loc_x)
                    else:
                        x_diff = tower_locations[0] - ego_loc_x
                    z_diff = tower_locations[1] - ego_loc_z
                    self.move_action(pred_ret, x_diff, z_diff, index=i)
            else:
                # If the hero cannot find a target whose distance from him is not zero, then take any action
                self.noop_action(pred_ret, index=i)

        return pred_ret

    def release_skill(self, action_space, target_idx, ids, index):
        # Based on the result of random initialization, choose a skill between normal attack and hero's four skills.
        # For example, suppose action_space[0][0][3:8] = [0.69794536 0.99747086 0.9519622  0.6940504  0.15969662]
        # The index, corresponding to the maximum value, is 4 and index=4 stands for skill 0.
        # Thus, the hero will try to launch skill 0.
        max_skill = max(action_space[index][0][3:8])
        max_idx = np.argwhere(action_space[index][0][3:8] == max_skill)[0][0]
        # if target is organ, normal attack (It will mask target organ if use skill)
        if ids[target_idx] == 2:
            max_idx = 3
        # Set the value corresponding to the chosen skill to maximum value in which_button dimension
        which_button_max = max(action_space[index][0][:13])
        action_space[index][0][max_idx] = which_button_max + 1

        self.skill_action(action_space, target_idx, ids, index=index)

        return

    def skill_action(self, action_space, target_idx, ids, index):
        """
        :param action_space:
        :param target_idx: Target's index in ids
        :param ids: A list which is used to discriminate different types of character. 0 stands for enemy hero; 1 stands for enemy soldier; 2 stands for enemy organ; 3 stands for monster.
        :param index: Hero's index in action space
        :return:
        """
        # In order to release skill, we have to set the value of offset_x, offset_z and target.
        # Not all of them are necessary and it depends on the type of the skill.
        # Here we don't distinguish them, since the unnecessary parameters won't be used.
        # You can refer to the instruction document of frame state for further information of the type of skills.

        # For the sake of simplicity, here we set offset_x and offset_z as (21, 21).
        # That means, the hero will release skill towards the target.

        offset_x, offset_z = 21, 21
        offset_x_max = max(action_space[index][0][38:80])
        offset_z_max = max(action_space[index][0][80:122])
        action_space[index][0][38 + offset_x] = offset_x_max + 1
        action_space[index][0][80 + offset_z] = offset_z_max + 1

        # First, find the index corresponding to the target in target dimension.
        # Second, set the value corresponding to the index as maximum in target dimension.
        target_max = max(action_space[index][0][122:])
        id = ids[target_idx]
        index_offset = target_idx - ids.index(id)

        if id == 0:
            # enemy hero
            index_offset = self.hero_idx_offset[index_offset]
            action_space[index][0][123 + index_offset] = target_max + 1
        elif id == 1:
            # soldier
            action_space[index][0][150 + index_offset] = target_max + 1
        elif id == 2:
            # organ
            action_space[index][0][160] = target_max + 1
        else:
            # monster
            # The index of monster in target dimension is saved in self.monster_idx_offset
            index_offset = self.monster_idx_offset[index_offset]
            action_space[index][0][130 + index_offset] = target_max + 1

        return

    def move_action(self, action_space, x_diff, z_diff, index):
        # Set the value corresponding to "move" action to maximum value in which_button dimension
        which_button_max = max(action_space[index][0][:13])
        action_space[index][0][2] = which_button_max + 1

        # Calculate the value in move dimension, which determines the direction of move action.
        # You can refer to the instruction document of action space for further information of the calculation method.
        if x_diff == 0:
            if z_diff > 0:
                theta = 90
            else:
                theta = 270

        elif x_diff > 0:
            theta = math.atan(z_diff / x_diff)
            theta = (theta / math.pi * 180 + 360) % 360
        else:
            theta = math.atan(z_diff / x_diff)
            theta = theta / math.pi * 180 + 180

        theta = (theta // 7.5 * 7.5 + 7.5) // 15 * 15
        theta_idx = int((12 - theta // 15) % 24 + 1)

        # Set the value corresponding to the move direction to maximum value.
        angle_max = max(action_space[index][0][13:38])
        action_space[index][0][13 + theta_idx] = angle_max + 1
        return

    def noop_action(self, action_space, index):
        which_button_max = max(action_space[index][0][:13])
        action_space[index][0][1] = which_button_max + 1
        return

    def get_ego_enemy(self, frame_state, runtime_ids):
        enemy_ids_2_idx, ego_ids_2_idx = {}, {}
        enemy_all_config_ids, ego_config_ids = [], []

        for idx, hero in enumerate(frame_state.hero_list):
            if hero.runtime_id in runtime_ids:
                self.ego_camp_id = hero.camp
                ego_config_ids.append(hero.config_id)
                ego_ids_2_idx[hero.config_id] = idx
            else:
                self.enemy_camp_id = hero.camp
                #  All characters still exist in frame state for a while even though they are dead.
                #  Thus, we had better confirm whether they are alive via hp(health point) value.
                enemy_all_config_ids.append(hero.config_id)
                if hero.hp > 0:
                    enemy_ids_2_idx[hero.config_id] = idx

        # The sequence of each hero's action space in pred_ret is the same as the sequence of each heroes' config_id in ascending order.
        ego_config_ids.sort()
        ego_camp = [ego_ids_2_idx[config_id] for config_id in ego_config_ids]

        enemy_all_config_ids.sort()
        enemy_camp = []
        self.hero_idx_offset = {}
        for action_index, config_id in enumerate(enemy_all_config_ids):
            idx = len(enemy_camp)
            self.hero_idx_offset[idx] = action_index
            if config_id in enemy_ids_2_idx:
                enemy_camp.append(enemy_ids_2_idx[config_id])

        self.monster_offset_dict = get_monster_offset_dict(
            self.ego_camp_id, self.enemy_camp_id
        )

        return ego_camp, enemy_camp

    def get_npc(self, frame_state):
        # Save the index of monster, enemy soldier and enemy organ.
        soldier_runtime_ids = []
        soldier_runtime_ids_2_idx = {}
        monster, monster_idx_offset = [], []
        organ = []
        for idx, npc in enumerate(frame_state.monster_list):
            # All characters still exist in frame state for a while even though they are dead.
            # Thus, we had better confirm whether they are alive via health point(hp) value.
            if npc.hp <= 0:
                continue
            monster.append(idx)
            offset = self.monster_config_id_2_idx(get_monster_camp(npc), npc.config_id)
            monster_idx_offset.append(offset)

        for idx, npc in enumerate(frame_state.soldier_list):
            if npc.hp <= 0:
                continue
            if npc.camp == self.enemy_camp_id:
                soldier_runtime_ids.append(npc.runtime_id)
                soldier_runtime_ids_2_idx[npc.runtime_id] = idx

        for idx, npc in enumerate(frame_state.organ_list):
            if npc.hp <= 0:
                continue

            if npc.camp == self.enemy_camp_id:
                organ.append(idx)

        # The sequence of soldier in target dimension is the same as soldiers' runtime_id in ascending order.
        soldier_runtime_ids.sort()
        # Only 10 indexes are left in target dimension and maybe there exists more than 10 soldiers in the current frame.
        # Thus, we choose ten of them.
        soldier_runtime_ids = soldier_runtime_ids[0:10]
        soldier = [
            soldier_runtime_ids_2_idx[runtime_id] for runtime_id in soldier_runtime_ids
        ]
        return soldier, monster, monster_idx_offset, organ

    def get_location(self, frame_state):
        # Save the location of enemy soldier, enemy hero, enemy organ and monster in a list.
        # Save their identity in a list as well. 0 stands fro enemy hero; 1 stands for enemy soldier; 2 stands for enemy organ; 3 stands for monster.
        location = []
        ids = []
        for enemy in self.enemy_camp:
            enemy_loc = frame_state.hero_list[enemy].location
            enemy_loc_x = enemy_loc.x
            enemy_loc_z = enemy_loc.z
            location.append([enemy_loc_x, enemy_loc_z])
            ids.append(0)

        for soldier in self.enemy_soldier:
            soldier_loc = frame_state.soldier_list[soldier].location
            soldier_loc_x = soldier_loc.x
            soldier_loc_z = soldier_loc.z
            location.append([soldier_loc_x, soldier_loc_z])
            ids.append(1)

        for organ in self.enemy_organ:
            organ_loc = frame_state.organ_list[organ].location
            organ_loc_x = organ_loc.x
            organ_loc_z = organ_loc.z
            location.append([organ_loc_x, organ_loc_z])
            ids.append(2)

        for monster in self.monster:
            monster_loc = frame_state.monster_list[monster].location
            monster_loc_x = monster_loc.x
            monster_loc_z = monster_loc.z
            location.append([monster_loc_x, monster_loc_z])
            ids.append(3)

        return ids, location

    def monster_config_id_2_idx(self, camp, config_id):
        return self.monster_offset_dict[config_id][camp]
