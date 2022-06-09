# -*- coding:utf-8 -*-
import json
import os
import sys

import numpy as np

if os.getenv("UPD_DICT") is not None and len(os.getenv("UPD_DICT")) > 0:
    update_dict = json.loads(os.getenv("UPD_DICT"))
else:
    update_dict = {}
from common_config import ModelConfig
from common_config import DimConfig


class Config:
    TRAIN_MODE = 0
    EVAL_MODE = 1
    BATTLE_MODE = 2
    AISERVERPORT = [10010, 10011]
    ALPHA = 0.5
    BETA = 0.01
    EPSILON = 1e-5
    INIT_CLIP_PARAM = 0.1
    BATCH_SIZE = 1

    GAMMA = 0.995
    LAMDA = 0.95
    STEPS = 128
    ENV_NAME = "kh-1v1"
    TASK_NAME = "test"
    ACTION_DIM = 79
    UPDATE_PATH = "../model/update"
    INIT_PATH = "../model/init"
    MEM_POOL_PATH = "./config/mem_pool.host_list"
    TASK_UUID = "123"
    IS_TRAIN = True
    SINGLE_TEST = False
    IS_CHECK = False
    ENEMY_TYPE = "network"
    EVAL_FREQ = 5


if __name__ == "__main__":
    print(
        np.sum(
            [
                [12944],
                [16],
                [16],
                [16],
                [16],
                [16],
                [16],
                [16],
                [16],
                [192],
                [256],
                [256],
                [256],
                [256],
                [128],
                [16],
                [16],
                [16],
                [16],
                [16],
                [16],
                [16],
                [512],
                [512],
            ]
        )
    )
