# -*- coding:utf-8 -*-
import numpy as np


class ModelConfig:
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 16
#    HERO_NUM = 3
    HERO_DATA_SPLIT_SHAPE = [
        4586,  # feature
        13,  # legal_action
        25,
        42,
        42,
        39,
        1,  # reward
        1,  # advantage
        1,  # action
        1,
        1,
        1,
        1,
        13,  # probs
        25,
        42,
        42,
        39,
        1,  # is_train
        1,  # sub_action
        1,
        1,
        1,
        1,
    ]
    HERO_SERI_VEC_SPLIT_SHAPE = [(6, 17, 17), (2852,)]
    HERO_FEATURE_IMG_CHANNEL = 6  # feature image channel for each hero
    HERO_LABEL_SIZE_LIST = [13, 25, 42, 42, 39]

    DIM_OF_SOLDIER_1_10 = [25] * 10
    DIM_OF_SOLDIER_11_20 = [25] * 10
    DIM_OF_ORGAN_1_3 = [29] * 3
    DIM_OF_ORGAN_4_6 = [29] * 3
    DIM_OF_MONSTER_1_20 = [28] * 20
    DIM_OF_HERO_FRD = [251] * 3
    DIM_OF_HERO_EMY = [251] * 3
    DIM_OF_HERO_MAIN = [44]
    DIM_OF_GLOBAL_INFO = [68]

    sample_one_size = np.sum(HERO_DATA_SPLIT_SHAPE)

    # tensorflow only
    use_xla = True
