import os
import random

import numpy as np
from hok import HoK1v1


def _generate_legal_action(env, states, common_ai):
    actions = []
    shapes = env.action_space()

    split_array = shapes.copy()[:-1]
    for i in range(1, len(split_array)):
        split_array[i] = split_array[i - 1] + split_array[i]

    for i in range(2):
        if common_ai[i]:
            actions.append(tuple([0] * 6))
            continue
        legal_action = np.split(states[i]["legal_action"], split_array)
        # print("legal_action", i, legal_action[0])
        act = []
        for j, _ in enumerate(shapes):
            tmp = []
            for k, la in enumerate(legal_action[j]):
                if la == 1:
                    tmp.append(k)
            a = random.randint(0, len(tmp) - 1)
            # print("for act id {}, avialiable action is {}".format(j, tmp))
            a = tmp[a]
            act.append(a)
            if j == 0:
                if legal_action[0][8]:
                    act[0] = 8
                    a = 8
                legal_action[5] = legal_action[5].reshape(-1, shapes[-1])[a]

        actions.append(tuple(act))
    return actions


def test_send_action(env, common_ai=None, eval=False, config_dicts=None):
    if config_dicts is None:
        config_dicts = [{"hero": "diaochan", "skill": "frenzy"} for _ in range(2)]
    print("======= test_send_action")
    print("try to get first state...", common_ai)
    obs, reward, done, state = env.reset(
        use_common_ai=common_ai, eval=eval, config_dicts=config_dicts, render=None
    )
    if common_ai[0]:
        print("first state: ", state[1].keys())
    else:
        print("first state: ", state[0].keys())
    i = 0
    print("first frame:", env.cur_frame_no)

    while True:
        print("----------------------run step ", i)
        actions = _generate_legal_action(env, state, common_ai)
        obs, reward, done, state = env.step(actions)
        if done[0] or done[1]:
            break
        i += 1
        # if i > 10:
        #     break
    env.close_game()
    print(state)


if __name__ == "__main__":
    CONFIG_PATH = "config.dat"
    GC_SERVER_ADDR = os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1:23432")
    # please replace the *ai_server_addr* with your ip address.
    AI_SERVER_ADDR = os.getenv("AI_SERVER_ADDR", "127.0.0.1")

    # remote gc server
    remote_mode = 2

    remote_param = {
        "remote_mode": remote_mode,
        "gc_server_addr": GC_SERVER_ADDR,
        "ai_server_addr": AI_SERVER_ADDR,
    }
    print(GC_SERVER_ADDR, AI_SERVER_ADDR)

    env = HoK1v1.load_game(
        runtime_id=0,
        game_log_path="./game_log",
        gamecore_path="./hok",
        config_path=CONFIG_PATH,
        eval_mode=False,
        remote_param=remote_param,
    )

    # test all 18 heros
    hero_list = [
        "luban",
        "miyue",
        "lvbu",
        "libai",
        "makeboluo",
        "direnjie",
        "guanyu",
        "diaochan",
        "luna",
        "hanxin",
        "huamulan",
        "buzhihuowu",
        "jvyoujing",
        "houyi",
        "zhongkui",
        "ganjiangmoye",
        "kai",
        "gongsunli",
        "peiqinhu",
        "shangguanwaner",
    ]
    # for i, h in enumerate(hero_list):
    #     print("=" * 15 + "test hero {}, {}/{}".format(h, i, len(hero_list)))
    test_send_action(
        env,
        common_ai=[False, False],
        eval=False,
        config_dicts=[{"hero": "diaochan", "skill": "frenzy"} for _ in range(2)],
    )
