# -*- coding:utf-8 -*-
import os


class Config:
    backend = os.getenv("AIARENA_BACKEND", "pytorch")
    actor_num = int(os.getenv("ACTOR_NUM", "1"))
    auto_bind_cpu = os.getenv("AUTO_BIND_CPU", "0") == "1"

    SEND_SAMPLE_FRAME = 963

    GAMMA = 0.995
    LAMDA = 0.95

    reward_config = {
        "whether_use_zero_sum_reward": 1,
        "team_spirit": 0,
        "time_scaling_discount": 1,
        "time_scaling_time": 4500,
        "reward_policy": {
            "hero_0": {
                "hp_rate_sqrt_sqrt": 1,
                "money": 0.001,
                "exp": 0.001,
                "tower": 1,
                "killCnt": 1,
                "deadCnt": -1,
                "assistCnt": 1,
                "total_hurt_to_hero": 0.1,
                "atk_monster": 0.1,
                "win_crystal": 1,
                "atk_crystal": 1,
            },
            "hero_1": {
                "hp_rate_sqrt_sqrt": 1,
                "money": 0.001,
                "exp": 0.001,
                "tower": 1,
                "killCnt": 1,
                "deadCnt": -1,
                "assistCnt": 1,
                "total_hurt_to_hero": 0.1,
                "atk_monster": 0.1,
                "win_crystal": 1,
                "atk_crystal": 1,
            },
            "hero_2": {
                "hp_rate_sqrt_sqrt": 1,
                "money": 0.001,
                "exp": 0.001,
                "tower": 1,
                "killCnt": 1,
                "deadCnt": -1,
                "assistCnt": 1,
                "total_hurt_to_hero": 0.1,
                "atk_monster": 0.1,
                "win_crystal": 1,
                "atk_crystal": 1,
            },
        },
        "policy_heroes": {
            "hero_0": [169, 112, 174],
            "hero_1": [176, 119, 157],
            "hero_2": [128, 163, 167],
        },
    }
