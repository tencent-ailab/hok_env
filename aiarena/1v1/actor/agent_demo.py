from numpy.random import rand
import numpy as np

from custom import Agent as BaseAgent
from rl_framework.common.logging import logger as LOG

pred_ret_shape = [(1, 84)] * 1
lstm_cell_shape = [(1, 1), (1, 1)]
tower_locations = [19820, 19820]
home_locations = [-x - 10000 for x in tower_locations]

from hok.hok1v1.lib.interface import (
    PLAYERCAMP_1,
    PLAYERCAMP_2,
    ACTOR_SOLDIER,
    ACTOR_TOWER,
    ACTOR_TOWER_HIGH,
    ACTOR_TOWER_SPRING,
    ACTOR_CRYSTAL,
)


class Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
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
        self.enemy_organ = []

    def process(self, state_dict, battle=False):
        state_dict = self.feature_post_process(state_dict)
        # Randomly initialize pred_ret
        pred_ret = []
        for shape in pred_ret_shape:
            pred_ret.append(rand(*shape).astype("float32"))
        lstm_info = []
        for shape in lstm_cell_shape:
            lstm_info.append(np.zeros(shape).astype("float32"))

        frame_state = state_dict["req_pb"]

        self._generate_rule_actions(frame_state, pred_ret, [state_dict["player_id"]])

        prob, action, d_action = self._sample_masked_action(
            pred_ret[0], state_dict["legal_action"]
        )
        value = np.zeros((1, 1))

        pred_ret2 = (prob, value, action, d_action)
        return d_action, d_action, self._sample_process(state_dict, pred_ret2)

    def _generate_rule_actions(self, frame_state, pred_ret, runtime_ids):
        # Get index of heroes, monsters, soldiers and organs in frame_state
        self.ego_camp, self.enemy_camp = self.get_ego_enemy(frame_state, runtime_ids)
        (
            self.enemy_soldier,
            #            self.monster,
            #            self.monster_idx_offset,
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
                dis_diff_hero2target = np.sqrt(x_diff ** 2 + z_diff ** 2)

                distances.append(dis_diff_hero2target)

            # Try to define if there is a target whose distance from the hero is not 0
            flags = np.array(distances) != 0

            if any(flags):
                # Get the distance and index of target who is nearest to hero
                min_dis = min(np.array(distances)[flags])
                target_idx = distances.index(min_dis)

                ego_hp_rate = (
                    frame_state.hero_list[hero_idx].hp
                    / frame_state.hero_list[hero_idx].max_hp
                )

                # If the minimal distance is less than the distance threshold, then launch an attack
                if min_dis < self.distance_threshold and ego_hp_rate > 0.5:
                    self.release_skill(pred_ret, target_idx, ids, index=i)
                else:
                    target_location_x = tower_locations[0]
                    target_location_z = tower_locations[1]

                    # go home
                    if ego_hp_rate <= 0.5:
                        target_location_x = home_locations[0]
                        target_location_z = home_locations[1]

                    # if the distance is too far, try to move towards enemy's organ
                    if self.ego_camp_id == PLAYERCAMP_2:
                        # We try to convert the red team's frame state data for symmetric processing so that we can calculate in the same way.
                        x_diff = target_location_x - (-ego_loc_x)
                        z_diff = target_location_z - (-ego_loc_z)
                    else:
                        x_diff = target_location_x - ego_loc_x
                        z_diff = target_location_z - ego_loc_z
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
        max_skill = max(action_space[index][0][3:11])
        max_idx = np.argwhere(action_space[index][0][3:11] == max_skill)[0][0]
        # if target is organ, normal attack (It will mask target organ if use skill)
        if ids[target_idx] == 2:
            max_idx = 3
        # Set the value corresponding to the chosen skill to maximum value in which_button dimension
        which_button_max = max(action_space[index][0][:10])
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

        offset_x, offset_z = 8, 8
        offset_x_max = max(action_space[index][0][44:60])
        offset_z_max = max(action_space[index][0][60:76])
        action_space[index][0][44 + offset_x] = offset_x_max + 1
        action_space[index][0][60 + offset_z] = offset_z_max + 1

        # First, find the index corresponding to the target in target dimension.
        # Second, set the value corresponding to the index as maximum in target dimension.
        target_max = max(action_space[index][0][60:])
        id = ids[target_idx]
        index_offset = target_idx - ids.index(id)

        if id == 0:
            # enemy hero
            index_offset = self.hero_idx_offset[index_offset]
            action_space[index][0][76 + 1 + index_offset] = target_max + 1
        elif id == 1:
            # soldier
            action_space[index][0][79 + index_offset] = target_max + 1
        elif id == 2:
            # organ
            action_space[index][0][83] = target_max + 1

        return

    def move_action(self, action_space, x_diff, z_diff, index):
        # Set the value corresponding to "move" action to maximum value in which_button dimension
        which_button_max = max(action_space[index][0][:12])

        action_space[index][0][2] = which_button_max + 1

        # Set the value corresponding to the move direction to maximum value.
        dis_diff_hero2target = np.sqrt(x_diff ** 2 + z_diff ** 2)
        move_x = x_diff / dis_diff_hero2target
        move_z = z_diff / dis_diff_hero2target

        while abs(move_x) * 2 < 8 and abs(move_z) * 2 < 8:
            move_x *= 2
            move_z *= 2

        move_x = int(move_x) + 8
        move_z = int(move_z) + 8

        move_x_max = max(action_space[index][0][12:28])
        move_z_max = max(action_space[index][0][28:44])

        action_space[index][0][12 + move_x] = move_x_max + 1
        action_space[index][0][28 + move_z] = move_z_max + 1
        return

    def noop_action(self, action_space, index):
        which_button_max = max(action_space[index][0][:12])
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

        return ego_camp, enemy_camp

    def get_npc(self, frame_state):
        # Save the index of monster, enemy soldier and enemy organ.
        soldier_runtime_ids = []
        soldier_runtime_ids_2_idx = {}
        # monster, monster_idx_offset = [], []
        organ = []
        # for idx, npc in enumerate(frame_state.monster_list):
        #    # All characters still exist in frame state for a while even though they are dead.
        #    # Thus, we had better confirm whether they are alive via health point(hp) value.
        #    if npc.hp <= 0:
        #        continue
        #    monster.append(idx)
        #    offset = self.monster_config_id_2_idx(get_monster_camp(npc), npc.config_id)
        #    monster_idx_offset.append(offset)

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
        soldier_runtime_ids = soldier_runtime_ids[0:4]
        soldier = [
            soldier_runtime_ids_2_idx[runtime_id] for runtime_id in soldier_runtime_ids
        ]
        # return soldier, monster, monster_idx_offset, organ
        return soldier, organ

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

        return ids, location
