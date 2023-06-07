# -*- coding: utf-8 -*-

from rl_framework.common.logging import log_time
import rl_framework.common.logging as LOG


class RewardManager:
    def __init__(self, gamma, lamda):
        self._gamma = gamma
        self._lamda = lamda
        self.advantage = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def reset(self):
        self.advantage = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    @log_time("reward_process")
    def calc_advantage(
        self,
        agent_id,
        hero_idx,
        final_reward,
        cur_value,
        next_value,
        all_hero_reward_detail,
        rl_info,
    ):
        frame_no = rl_info.frame_no
        # calc my final reward
        my_final_reward = self.calc_final_reward(hero_idx, all_hero_reward_detail)

        # calc advantage and reward_sum
        # advantage: used for policy loss
        # reward_sum: used for value loss
        delta = -cur_value + final_reward + self._gamma * next_value
        self.advantage[agent_id][hero_idx] = (
            self.advantage[agent_id][hero_idx] * self._gamma * self._lamda + delta
        )
        reward_sum = self.advantage[agent_id][hero_idx] + cur_value

        LOG.debug(
            "hero_idx:%d final_reward:%f advantage:%f reward_sum:%f frame_no:%f"
            % (
                hero_idx,
                final_reward,
                self.advantage[agent_id][hero_idx],
                reward_sum,
                frame_no,
            )
        )

        return self.advantage[agent_id][hero_idx], reward_sum

    def calc_final_reward(self, hero_idx, all_hero_reward_detail):
        """
        all_hero_reward_detail = [
            {"money":0.0, ...},    # ally_hero_0
            {"money":0.0, ...},    # ally_hero_1
            {"money":0.0, ...},    # ally_hero_2
            {"money":0.0, ...},    # emeny_hero_0
            {"money":0.0, ...},    # enemy_hero_1
            {"money":0.0, ...},    # enemy_hero_2
        ]

        """

        final_reward = 0.0
        if all_hero_reward_detail is None:
            LOG.error("all_hero_reward_detail is None")
            return final_reward
        for key in all_hero_reward_detail[hero_idx]:
            value = all_hero_reward_detail[hero_idx][key]
            final_reward += value
            LOG.debug(
                "hero_idx:%d reward_name:%s reward_value:%f" % (hero_idx, key, value)
            )

        LOG.debug("hero_idx:%d my_final_reward:%f" % (hero_idx, final_reward))

        return final_reward
