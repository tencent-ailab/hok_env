import os


class Config:
    slow_time = float(os.getenv("SLOW_TIME", "0").strip())
    backend = os.getenv("AIARENA_BACKEND", "pytorch")
    use_init_model = os.getenv("AIARENA_USE_INIT_MODEL", "0") == "1"
    init_model_path = os.getenv(
        "AIARENA_INIT_MODEL_PATH", "/aiarena/code/learner/model/init/"
    )
    load_optimizer_state = os.getenv("AIARENA_LOAD_OPTIMIZER_STATE", "1") == "1"

    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 16
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

    HERO_NUM = 3
    HERO_IS_REINFORCE_TASK_LIST = [[True] * len(HERO_LABEL_SIZE_LIST)] * HERO_NUM
    INIT_LEARNING_RATE_START = 0.0006
    BETA_START = 0.008
    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    data_shapes = [
        [sum(HERO_DATA_SPLIT_SHAPE) * LSTM_TIME_STEPS + LSTM_UNIT_SIZE * 2]
    ] * HERO_NUM
