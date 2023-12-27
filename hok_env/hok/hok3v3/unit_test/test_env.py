import os
from numpy.random import rand
import numpy as np
from hok.hok3v3.lib import lib3v3 as interface
from hok.hok3v3.hero_config import get_default_hero_config, interface_default_config
from hok.hok3v3.server import AIServer
from hok.common.gamecore_client import GamecoreClient
from hok.hok3v3.env import Environment
from hok.common.log import setup_logger
from hok.common.log import logger as LOG


def get_hok3v3(GC_SERVER_ADDR):
    LOG.info(f"init hok: {interface_default_config} {GC_SERVER_ADDR}")
    lib_processor = interface.Interface()
    lib_processor.Init(interface_default_config)

    aiservers = []
    for i in range(2):
        addr = f"tcp://0.0.0.0:{35150 + i}"
        aiservers.append(AIServer(addr, lib_processor))

    game_launcher = GamecoreClient(
        server_addr=GC_SERVER_ADDR,
        gamecore_req_timeout=30000,
        default_hero_config=get_default_hero_config(),
        max_frame_num=20000,
    )

    env = Environment(
        aiservers,
        lib_processor,
        game_launcher,
        runtime_id="test-env",
    )
    return env


def random_predict(features, req_pb):
    pred_ret_shape = [(1, 162), (1, 162), (1, 162)]
    lstm_cell_shape = [(1, 16), (1, 16)]
    pred_ret = []
    for shape in pred_ret_shape:
        pred_ret.append(rand(*shape).astype("float32"))
    lstm_info = []
    for shape in lstm_cell_shape:
        lstm_info.append(np.zeros(shape).astype("float32"))
    return pred_ret, lstm_info


def run_test():
    setup_logger()
    GC_SERVER_ADDR = os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1:23432")
    # please replace the *AI_SERVER_ADDR* with your ip address.
    AI_SERVER_ADDR = os.getenv("AI_SERVER_ADDR", "127.0.0.1")

    LOG.info(GC_SERVER_ADDR, AI_SERVER_ADDR)
    env = get_hok3v3(GC_SERVER_ADDR)

    use_common_ai = [True, False]
    camp_config = {
        "mode": "3v3",
        "heroes": [
            [{"hero_id": 190}, {"hero_id": 173}, {"hero_id": 117}],
            [{"hero_id": 141}, {"hero_id": 111}, {"hero_id": 107}],
        ],
    }
    env.reset(use_common_ai, camp_config, eval_mode=True)

    gameover = False
    cnt = 0
    while not gameover and cnt <= 200:
        if cnt % 100 == 0:
            LOG.info(f"----------------------run step {cnt}")
        cnt += 1

        for i, is_comon_ai in enumerate(use_common_ai):
            if is_comon_ai:
                continue

            not_skip_prediction, features, p_game_data = env.step_feature(i)
            req_pb = p_game_data.frame_state

            if not_skip_prediction:
                probs, lstm_info = random_predict(features, req_pb)
                sample = env.step_action(i, features, probs, p_game_data, lstm_info)

            gameover = req_pb.gameover

    env.close_game(force=True)


if __name__ == "__main__":
    run_test()
