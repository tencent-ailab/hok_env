import os
import random

import numpy as np
from hok.hok1v1 import HoK1v1


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


def test_send_action(env, common_ai, eval, camp_config):
    print("======= test_send_action")
    print("camp_config", camp_config)
    print("common_ai", common_ai)
    print("try to get first state...")
    obs, reward, done, state = env.reset(
        camp_config, use_common_ai=common_ai, eval=eval
    )
    if common_ai[0]:
        print("first state: ", state[1].keys())
    else:
        print("first state: ", state[0].keys())
    i = 0
    print("first frame:", env.cur_frame_no)

    while True:
        if i % 100 == 0:
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
    from hok.common.gamecore_client import GamecoreClient
    from hok.hok1v1.env1v1 import interface_default_config
    from hok.hok1v1.hero_config import get_default_hero_config
    import hok.hok1v1.lib.interface as interface

    lib_processor = interface.Interface()
    lib_processor.Init(interface_default_config)

    # please replace the *AI_SERVER_ADDR* with your ip address.
    GC_SERVER_ADDR = os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1:23432")
    AI_SERVER_ADDR = os.getenv("AI_SERVER_ADDR", "127.0.0.1")
    gamecore_req_timeout = 3000

    print(GC_SERVER_ADDR, AI_SERVER_ADDR)

    AGENT_NUM = 2
    addrs = []
    for i in range(AGENT_NUM):
        addrs.append("tcp://0.0.0.0:{}".format(35150 + i))

    game_launcher = GamecoreClient(
        server_addr=GC_SERVER_ADDR,
        gamecore_req_timeout=gamecore_req_timeout,
        default_hero_config=get_default_hero_config(),
    )

    env = HoK1v1(
        "test-env",
        game_launcher,
        lib_processor,
        addrs,
        aiserver_ip=AI_SERVER_ADDR,
    )

    from hok.common.camp import HERO_DICT, camp_iterator_1v1_roundrobin_camp_heroes

    camp_iter = camp_iterator_1v1_roundrobin_camp_heroes(HERO_DICT.values())
    camp_config = next(camp_iter)

    test_send_action(env, common_ai=[False, True], eval=False, camp_config=camp_config)
