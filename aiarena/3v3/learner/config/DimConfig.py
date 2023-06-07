# coding:utf-8

class DimConfig:

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
