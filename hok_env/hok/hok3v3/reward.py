from collections import defaultdict

from hok.common.log import logger as LOG


def merge_dicts(a, b):
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]


class RewardConfig:
    default_reward_policy = {
        "hp_rate_sqrt": 1,
        "money": 0.001,
        "exp": 0.001,
        "tower": 1,
        "killCnt": 1,
        "deadCnt": -1,
        "assistCnt": 1,
        "total_hurt_to_hero": 0.1,
        "ep_rate": 0.1,
        "win_crystal": 1,
    }
    default_reward_config = {
        "whether_use_zero_sum_reward": 1,
        "team_spirit": 0.2,
        "time_scaling_discount": 1,
        "time_scaling_time": 4500,
        "reward_policy": {
            "policy_name_0": default_reward_policy.copy(),
        },
        # bind hero_id to policy: hero_id to policy_name
        "hero_policy": {
            1: "policy_name_0",
        },
        # bind hero_id to policy: policy_name to hero_id, use hero_policy first
        "policy_heroes": {"policy_name_0": [1, 2]},
    }

    def __init__(self, reward_config=None) -> None:
        _reward_config = self.default_reward_config.copy()
        if reward_config is not None:
            merge_dicts(_reward_config, reward_config)

        self.hero_policy = _reward_config["hero_policy"]
        self.policy_heroes = _reward_config["policy_heroes"]
        self.reward_policy = _reward_config["reward_policy"]

        self.whether_use_zero_sum_reward = _reward_config.get(
            "whether_use_zero_sum_reward", -1
        )
        self.team_spirit = _reward_config.get("team_spirit", -1)
        self.time_scaling_discount = _reward_config.get("time_scaling_discount", -1)
        self.time_scaling_time = _reward_config.get("time_scaling_time", -1)

    def set_reward_policy(self, policy_name, reward_item_config):
        self.reward_policy[policy_name] = reward_item_config

    def set_hero_policy(self, policy_name: str, hero_id: int):
        if policy_name not in self.reward_policy:
            raise Exception("Unknown policy_name: {}".format(policy_name))

        self.hero_policy[hero_id] = policy_name

    def update_reward_config(self, libprocessor):
        """
        update reward_config in libprocessor
        """
        for hero_id, reward_policy_name in self.hero_policy.items():
            reward_policy = self.reward_policy[reward_policy_name]
            self._update_reward_policy(libprocessor, hero_id, reward_policy)

        for reward_policy_name, heroes in self.policy_heroes.items():
            reward_policy = self.reward_policy[reward_policy_name]

            for hero_id in heroes:
                if hero_id not in self.hero_policy:
                    self._update_reward_policy(libprocessor, hero_id, reward_policy)

            libprocessor.SetTimeScalingTime(self.time_scaling_time)
            libprocessor.SetTimeScalingDiscount(self.time_scaling_discount)
            libprocessor.SetTeamSpirit(self.team_spirit)
            libprocessor.SetWhetherUseZeroSumReward(self.whether_use_zero_sum_reward)

    def _update_reward_policy(self, libprocessor, hero_id, reward_policy):
        for reward_name, reward_weight in reward_policy.items():
            libprocessor.SetHeroRewardWeight(hero_id, reward_name, reward_weight)

    def get_reward_policy(self, hero_id):
        if hero_id in self.hero_policy:
            reward_policy_name = self.hero_policy[hero_id]
            return self.reward_policy[reward_policy_name]

        for reward_policy_name, heroes in self.policy_heroes.items():
            if hero_id in heroes:
                return self.reward_policy[reward_policy_name]
        return {}

    def get_configured_hero_id(self):
        hero_id_dict = {}
        for hero_id, _ in self.hero_policy.items():
            hero_id_dict[hero_id] = 1
        for _, heroes in self.policy_heroes.items():
            for hero_id in heroes:
                hero_id_dict[hero_id] = 1
        return hero_id_dict.keys()


def update_reward_config(libprocessor, reward_config):
    reward_config = RewardConfig(reward_config)
    reward_config.update_reward_config(libprocessor)

    LOG.info(
        "Update reward config: time_scaling_time:{}, time_scaling_discount:{}, team_spirit:{}, whether_use_zero_sum_reward:{}",
        reward_config.time_scaling_time,
        reward_config.time_scaling_discount,
        reward_config.team_spirit,
        reward_config.whether_use_zero_sum_reward,
    )
    for hero_id in reward_config.get_configured_hero_id():
        LOG.info(
            "Update hero reward config: {} -> {}",
            hero_id,
            reward_config.get_reward_policy(hero_id),
        )


if __name__ == "__main__":

    class FakeLibProcessor:
        def __init__(self) -> None:
            self.reward_config = defaultdict(lambda: defaultdict(float))

        def SetTimeScalingTime(self, time_scaling_time: float):
            self.time_scaling_time = time_scaling_time

        def SetTimeScalingDiscount(self, time_scaling_discount: float):
            self.time_scaling_discount = time_scaling_discount

        def SetTeamSpirit(self, team_spirit: float):
            self.team_spirit = team_spirit

        def SetWhetherUseZeroSumReward(self, whether_use_zero_sum_reward: int):
            self.whether_use_zero_sum_reward = whether_use_zero_sum_reward

        def SetHeroRewardWeight(
            self, config_id: int, reward_name: str, reward_weight: float
        ):
            self.reward_config[config_id][reward_name] = reward_weight

        def get_reward_policy(self, hero_id):
            return self.reward_config.get(hero_id, {})

    libprocessor = FakeLibProcessor()
    reward_config = RewardConfig()
    reward_config.update_reward_config(libprocessor)

    assert reward_config.time_scaling_time == libprocessor.time_scaling_time
    assert reward_config.time_scaling_discount == libprocessor.time_scaling_discount
    assert reward_config.team_spirit == libprocessor.team_spirit
    assert (
        reward_config.whether_use_zero_sum_reward
        == libprocessor.whether_use_zero_sum_reward
    )

    for hero_id, reward_policy in libprocessor.reward_config.items():
        assert reward_policy == reward_config.get_reward_policy(hero_id)

    for hero_id in reward_config.get_configured_hero_id():
        assert libprocessor.get_reward_policy(
            hero_id
        ) == reward_config.get_reward_policy(hero_id)
    print(RewardConfig.default_reward_config)
