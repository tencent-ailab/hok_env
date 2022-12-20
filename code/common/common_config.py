import os


class DimConfig:
    # main camp soldier
    DIM_OF_SOLDIER_1_10 = [18, 18, 18, 18]
    # enemy camp soldier
    DIM_OF_SOLDIER_11_20 = [18, 18, 18, 18]
    # main camp organ
    DIM_OF_ORGAN_1_2 = [18, 18]
    # enemy camp organ
    DIM_OF_ORGAN_3_4 = [18, 18]
    # main camp hero
    DIM_OF_HERO_FRD = [235]
    # enemy camp hero
    DIM_OF_HERO_EMY = [235]
    # public hero info
    DIM_OF_HERO_MAIN = [14]  # main_hero_vec

    DIM_OF_GLOBAL_INFO = [25]


class ModelConfig:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        809,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        8,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        512,
        512,
    ]
    SERI_VEC_SPLIT_SHAPE = [(725,), (84,)]
    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]  # means each task whether need reinforce

    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001
    TASK_ID = 15428
    TASK_UUID = "a2dbb49f-8a67-4bd4-9dc5-69e78422e72e"

    BATCH_SIZE = 512
    TARGET_EMBED_DIM = 32

    data_keys = (
        "observation,reward,advantage,"
        "label0,label1,label2,label3,label4,label5,"
        "prob0,prob1,prob2,prob3,prob4,prob5,"
        "weight0,weight1,weight2,weight3,weight4,weight5,"
        "is_train, lstm_cell, lstm_hidden_state"
    )
    data_shapes = [
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
    key_types = (
        "tf.float32,tf.float32,tf.float32,"
        "tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,"
        "tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,"
        "tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,"
        "tf.float32,tf.float32,tf.float32,tf.float32"
    )

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]


class Config:
    slow_time = 0.0
    TRAIN_MODE = 0
    EVAL_MODE = 1
    BATTLE_MODE = 2
    AISERVERPORT = [10010, 10011]
    # kinghonour: 1e-5 atari: 2.5e-4
    INIT_LEARNING_RATE = 1e-4
    END_LEARNING_RATE = 1e-5
    ALPHA = 0.5
    BETA = 0.01
    EPSILON = 1e-5
    INIT_CLIP_PARAM = 0.1
    # kinghonour:4096 atari:256
    BATCH_SIZE = 4096
    EPISODE = 20000000
    GAMMA = 0.995
    LAMDA = 0.95
    STEPS = 128
    EPOCHES = 4
    MINI_BATCH_NUM = 4
    ENV_NAME = "BowlingNoFrameskip-v4"
    MIN_POLICY = 0.00005
    T = 1
    TASK_NAME = "test"
    MEM_PROCESS_NUM = 8
    DATA_KEYS = "input_data"
    KEY_TYPES = "tf.float32"
    ACTION_DIM = 79
    SERVER_PORT = 30166
    ACTOR_NUM = 0
    LEARNER_NUM = 0
    EACH_LEARNER_NUM = 0
    PARAMS_PATH = "/code/gpu_code/learner/model/update"
    GPU_SERVER_LIST = ""
    UPDATE_PATH = "../model/update"
    INIT_PATH = "../model/init"
    MEM_POOL_PATH = "./config/mem_pool.host_list"
    TASK_UUID = "123"
    IS_TRAIN = True
    SINGLE_TEST = False
    IS_CHECK = False

    ENEMY_TYPE = "network"
    if os.getenv("ENEMY_TYPE") is not None:
        enemy_type = int(os.getenv("ENEMY_TYPE"))
        if enemy_type == 0:
            ENEMY_TYPE = "random"
        elif enemy_type == 1:
            ENEMY_TYPE = "common_ai"
        elif enemy_type == 2:
            ENEMY_TYPE = "network"
    ENV_RULE = "none"
    EVAL_FREQ = 5
    # if os.getenv("SLOW_TIME") is not None:
    #     slow_time = float(os.getenv("SLOW_TIME").strip())
