# -*- coding:utf-8 -*-
import os


class ModelConfig:
    backend = "pytorch"
    NETWORK_NAME = "network"
    HERO_NUM = 3
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 16
    TARGET_EMBEDDING_DIM = 128
    VALUE_HEAD_NUM = 1
    HERO_DATA_SPLIT_SHAPE = [
        [
            4586,
            13,
            25,
            42,
            42,
            39,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            13,
            25,
            42,
            42,
            39,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        [
            4586,
            13,
            25,
            42,
            42,
            39,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            13,
            25,
            42,
            42,
            39,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        [
            4586,
            13,
            25,
            42,
            42,
            39,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            13,
            25,
            42,
            42,
            39,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    ]

    HERO_INFER_DATA_SPLIT_SHAPE = [[4586, 1676], [4586, 1676], [4586, 1676]]

    HERO_LEGAL_DIM = [13, 25, 42, 42, 39]
    HERO_SERI_VEC_SPLIT_SHAPE = [
        [(6, 17, 17), (2852,)],
        [(6, 17, 17), (2852,)],
        [(6, 17, 17), (2852,)],
    ]
    HERO_FEATURE_IMG_CHANNEL = [[6], [6], [6]]  # feature image channel for each hero
    HERO_LABEL_SIZE_LIST = [
        [13, 25, 42, 42, 39],
        [13, 25, 42, 42, 39],
        [13, 25, 42, 42, 39],
    ]
    HERO_IS_REINFORCE_TASK_LIST = [
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
    ]
    HERO_NEED_REINFORCE_PARAM_BUTTON_LABEL_LIST = [
        [2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8],
    ]

    HERO_DATA_ORDER_DICT = {
        128: 0,
        126: 0,
        134: 0,
        144: 0,
        193: 0,
        522: 0,
        518: 0,
        503: 0,
        154: 0,
        140: 0,
        178: 0,
        180: 0,
        507: 0,
        123: 0,
        143: 0,
        166: 0,
        510: 0,  # topsolo
        183: 1,
        116: 1,
        162: 1,
        131: 1,
        146: 1,
        163: 1,
        130: 1,
        107: 1,
        511: 1,
        167: 1,
        150: 1,
        153: 1,
        506: 1,
        129: 1,
        117: 1,
        195: 1,
        502: 1,  # jungle
        136: 2,
        182: 2,
        152: 2,
        141: 2,
        157: 2,
        513: 2,
        182: 2,
        156: 2,
        142: 2,
        108: 2,
        124: 2,
        137: 2,
        110: 2,
        127: 2,
        190: 2,
        106: 2,  # midsolo
        133: 3,
        174: 3,
        132: 3,
        111: 3,
        199: 3,
        173: 3,
        196: 3,
        169: 3,
        177: 3,
        192: 3,
        112: 3,
        508: 3,  # adc
        171: 4,
        168: 4,
        187: 4,
        175: 4,
        194: 4,
        186: 4,
        114: 4,
        118: 4,
        189: 4,
        505: 4,
        113: 4,
        120: 4,
        135: 4,
        184: 4,
        501: 4,  # support
    }

    INIT_LEARNING_RATE_START = 0.0006
    BETA_START = 0.008
    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001
    TASK_ID = int(os.getenv("TASK_ID", "16980"))
    TASK_UUID = os.getenv("TASK_UUID", "10ad0318-893f-4426-ac8e-44f109561350")
    data_keys = "hero1_data,hero2_data,hero3_data"
    # data_shapes = [[205488], [205488], [205488], [205488], [205488]]
    data_shapes = [
        [sum(HERO_DATA_SPLIT_SHAPE[0]) * LSTM_TIME_STEPS + LSTM_UNIT_SIZE * 2]
    ] * 3
    # data_shapes = [[sum(HERO_DATA_SPLIT_SHAPE[0])*LSTM_TIME_STEPS ]] * 3
    # key_types = "tf.float32,tf.float32,tf.float32,tf.float32,tf.float32"
    LABEL_ACTION_NUM = "10,25,9,61,61,61,61,61"

    # class DimConfig:

    KEYS_OF_VEC = [
        "frd_hero_vec_1",
        "frd_hero_vec_2",
        "frd_hero_vec_3",
        "emy_hero_vec_1",
        "emy_hero_vec_2",
        "emy_hero_vec_3",
        "main_hero_vec",
        "soldier_1",
        "soldier_2",
        "soldier_3",
        "soldier_4",
        "soldier_5",
        "soldier_6",
        "soldier_7",
        "soldier_8",
        "soldier_9",
        "soldier_10",
        "soldier_11",
        "soldier_12",
        "soldier_13",
        "soldier_14",
        "soldier_15",
        "soldier_16",
        "soldier_17",
        "soldier_18",
        "soldier_19",
        "soldier_20",
        "organ_1",
        "organ_2",
        "organ_3",
        "organ_4",
        "organ_5",
        "organ_6",
        "monster_1",
        "monster_2",
        "monster_3",
        "monster_4",
        "monster_5",
        "monster_6",
        "monster_7",
        "monster_8",
        "monster_9",
        "monster_10",
        "monster_11",
        "monster_12",
        "monster_13",
        "monster_14",
        "monster_15",
        "monster_16",
        "monster_17",
        "monster_18",
        "monster_19",
        "monster_20",
    ]

    KEYS_OF_SOLDIER_1_10 = [
        "soldier_1",
        "soldier_2",
        "soldier_3",
        "soldier_4",
        "soldier_5",
        "soldier_6",
        "soldier_7",
        "soldier_8",
        "soldier_9",
        "soldier_10",
    ]
    DIM_OF_SOLDIER_1_10 = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25]

    KEYS_OF_SOLDIER_11_20 = [
        "soldier_11",
        "soldier_12",
        "soldier_13",
        "soldier_14",
        "soldier_15",
        "soldier_16",
        "soldier_17",
        "soldier_18",
        "soldier_19",
        "soldier_20",
    ]
    DIM_OF_SOLDIER_11_20 = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25]

    KEYS_OF_ORGAN_1_3 = ["organ_1", "organ_2", "organ_3"]
    KEYS_OF_ORGAN_4_6 = ["organ_4", "organ_5", "organ_6"]
    DIM_OF_ORGAN_1_3 = [29, 29, 29]

    DIM_OF_ORGAN_4_6 = [29, 29, 29]

    KEYS_OF_MONSTER_1_20 = [
        "monster_1",
        "monster_2",
        "monster_3",
        "monster_4",
        "monster_5",
        "monster_6",
        "monster_7",
        "monster_8",
        "monster_9",
        "monster_10",
        "monster_11",
        "monster_12",
        "monster_13",
        "monster_14",
        "monster_15",
        "monster_16",
        "monster_17",
        "monster_18",
        "monster_19",
        "monster_20",
    ]
    DIM_OF_MONSTER_1_20 = [
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
        28,
    ]

    KEYS_OF_HERO_MAIN = ["main_hero_vec"]
    KEYS_OF_HERO_FRD = ["frd_hero_vec_1", "frd_hero_vec_2", "frd_hero_vec_3"]
    KEYS_OF_HERO_EMY = ["emy_hero_vec_1", "emy_hero_vec_2", "emy_hero_vec_3"]
    # hero
    # 394 + 14
    DIM_OF_HERO_FRD = [251, 251, 251]

    DIM_OF_HERO_EMY = [251, 251, 251]

    DIM_OF_HERO_MAIN = [44]  # main_hero_vec

    DIM_OF_GLOBAL_INFO = [68]

    ### Embbeding ###
    NUM_OF_HEROS = 6
    # Skill
    DIM_OF_Skill_EMBEDDED = 32
    NUM_OF_Skill_OF_ONE_HERO = 8  # 7 add one active-equip
    # Skill_DICT_SIZE = 1976 + 1 # v45
    Skill_DICT_SIZE = 2116 + 1  # v51
    Skill_state_dim = 5
    DIM_OF_SKILL_FRD = [5, 5, 5]

    DIM_OF_SKILL_EMY = [5, 5, 5]
    # Buff
    DIM_OF_BUFFSkill_EMBEDDED = 32
    NUM_OF_BUFFSkill_OF_ONE_HERO = 10
    BUFFSkill_DICT_SIZE = 5749 + 1
    # Equipment
    DIM_OF_EQUIPMENT_EMBEDDED = 32
    NUM_OF_EQUIPMENT_OF_ONE_HERO = 6
    # EQUIP_DICT_SIZE = 95 + 1 # v45
    EQUIP_DICT_SIZE = 98 + 1  # v51 add 3 new equip


