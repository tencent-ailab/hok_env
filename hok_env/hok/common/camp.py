import random
import os
import json
import base64
import itertools
import threading

HERO_DICT = {
    "xiaoqiao": 106,
    "zhaoyun": 107,
    "mozi": 108,
    "sunshangxiang": 111,
    "luban": 112,
    "zhongwuyan": 117,
    "bianque": 119,
    "baiqi": 120,
    "miyue": 121,
    "lvbu": 123,
    "caocao": 128,
    "gongbenwuzang": 130,
    "libai": 131,
    "makeboluo": 132,
    "direnjie": 133,
    "xiangyu": 135,
    "guanyu": 140,
    "diaochan": 141,
    "luna": 146,
    "hanxin": 150,
    "wangzhaojun": 152,
    "huamulan": 154,
    "ailin": 155,
    "buzhihuowu": 157,
    "jvyoujing": 163,
    "sunwukong": 167,
    "houyi": 169,
    "liyuanfang": 173,
    "yangyuhuan": 176,
    "yuji": 174,
    "zhongkui": 175,
    "ganjiangmoye": 182,
    "guiguzi": 189,
    "zhugeliang": 190,
    "huangzhong": 192,
    "kai": 193,
    "sulie": 194,
    "bailishouyue": 196,
    "gongsunli": 199,
    "peiqinhu": 502,
    "sunce": 510,
    "yao": 522,
    "shangguanwaner": 513,
}


class GameMode:
    G1v1 = "1v1"
    G3v3 = "3v3"
    G5v5dld = "5v5dld"


def thread_safe_iterator(iterator):
    lock = threading.Lock()
    while True:
        try:
            with lock:
                value = next(iterator)
        except StopIteration:
            return
        yield value


# 循环返回camps, 每次大循环前对camps进行shuffle
def _camp_iterator_shuffle_cycle(mode, camps):
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield {
                "mode": mode,
                "heroes": camp,
            }


# 循环列表里的固定英雄阵容
def camp_iterator_cycle_camps(mode="1v1", camps=None):
    if camps is None:
        camps = [
            [
                [{"hero_id": 112, "skill_id": 80115, "symbol": [1512, 1512]}],
                [{"hero_id": 121, "skill_id": 80115, "symbol": [1512, 1512]}],
            ]
        ]
    return _camp_iterator_shuffle_cycle(mode, camps)


# 多英雄阵容两两组合循环
def camp_iterator_roundrobin_camp_heroes(mode=GameMode.G1v1, camp_heroes=None):
    if camp_heroes is None:
        camp_heroes = [
            [{"hero_id": 112, "skill_id": 80115, "symbol": [1512, 1512]}],
            [{"hero_id": 121, "skill_id": 80115, "symbol": [1512, 1512]}],
        ]
    camps = [list(x) for x in itertools.product(camp_heroes, camp_heroes)]
    return _camp_iterator_shuffle_cycle(mode, camps)


# 1v1 按英雄列表循环对战
def camp_iterator_1v1_roundrobin_camp_heroes(hero_ids=None):
    if hero_ids is None:
        hero_ids = [111]

    return camp_iterator_roundrobin_camp_heroes(
        GameMode.G1v1, [[{"hero_id": hero_id}] for hero_id in hero_ids]
    )


# 3v3 按英雄列表循环对战（候选阵容）
def camp_iterator_3v3_roundrobin_camp_heroes(camp_hero_ids=None):
    if camp_hero_ids is None:
        camp_hero_ids = [
            [157, 174, 167],  # 不知火舞、虞姬、孙悟空 —— 比较全面的阵容缺点没有硬控 (高爆法有POKE)
        ]

    return camp_iterator_roundrobin_camp_heroes(
        GameMode.G3v3,
        [[{"hero_id": hero_id} for hero_id in hero_ids] for hero_ids in camp_hero_ids],
    )


# 5v5大乱斗 按英雄列表循环对战（候选阵容）
def camp_iterator_5v5dld_roundrobin_camp_heroes(camp_hero_ids=None):
    if camp_hero_ids is None:
        camp_hero_ids = [
            [112, 121, 123, 131, 132],
        ]

    return camp_iterator_roundrobin_camp_heroes(
        GameMode.G5v5dld,
        [[{"hero_id": hero_id} for hero_id in hero_ids] for hero_ids in camp_hero_ids],
    )


# 按分路选择一个英雄出场, 所有阵容循环对战
def camp_iterator_roundrobin_lane_heroes(mode=GameMode.G3v3, lane_hero_ids=None):
    if lane_hero_ids is None:
        lane_hero_ids = [[190, 141], [173, 111], [107, 117]]

    camp_heroes = []
    for camp_heroes_one in itertools.product(*lane_hero_ids):
        camp_heroes.append([{"hero_id": hero_id} for hero_id in camp_heroes_one])
    return camp_iterator_roundrobin_camp_heroes(mode, camp_heroes)


# 3v3按分路选择一个英雄出场, 所有阵容循环对战
def camp_iterator_3v3_roundrobin_lane_heroes(lane_hero_ids=None):
    if lane_hero_ids is None:
        lane_hero_ids = [[190, 141], [173, 111], [107, 117]]

    return camp_iterator_roundrobin_lane_heroes(GameMode.G3v3, lane_hero_ids)


def _get_default_config_str(default_mode):
    if default_mode == GameMode.G3v3:
        default_config_str = base64.b64encode(
            b'[[157, 174, 167]]',
        ).decode()
    elif default_mode == GameMode.G5v5dld:
        default_config_str = base64.b64encode(
            b"[[112, 121, 123, 131, 132]]"
        ).decode()
    else:
        default_config_str = base64.b64encode(
            # 1v1 多英雄，以下为第三届初赛配置（共 5 种英雄）
            b'[112,169,133,199,132]',
        ).decode()
    return default_config_str


def _get_default_driver(default_mode):
    if default_mode == GameMode.G3v3:
        default_driver = "3v3_roundrobin_camp_heroes"
    elif default_mode == GameMode.G5v5dld:
        default_driver = "5v5dld_roundrobin_camp_heroes"
    else:
        default_driver = "1v1_roundrobin_camp_heroes"
    return default_driver


# 创建阵容配置的迭代器用于创建对战
# 当两个参数为空的时候, 从环境变量获取对应的参数
# 当相关环境变量未设置时使用默认值
def camp_iterator(
    driver=None,
    driver_config=None,
    default_mode=os.getenv("CAMP_DEFAULT_MODE", GameMode.G1v1),
):
    if driver is None:
        driver = os.getenv("CAMP_DRIVER", _get_default_driver(default_mode))

    if driver_config is None:
        driver_config_str = os.getenv(
            "CAMP_DRIVER_CONFIG", _get_default_config_str(default_mode)
        )
        try:
            driver_config_str = base64.b64decode(driver_config_str).decode()
        except Exception:
            pass
        driver_config = json.loads(driver_config_str)

    if driver == "roundrobin_camp_heroes":
        return camp_iterator_roundrobin_camp_heroes(**driver_config)
    elif driver == "cycle_camps":
        return camp_iterator_cycle_camps(**driver_config)
    elif driver == "roundrobin_lane_heroes":
        return camp_iterator_roundrobin_lane_heroes(**driver_config)
    elif driver == "1v1_roundrobin_camp_heroes":
        return camp_iterator_1v1_roundrobin_camp_heroes(driver_config)
    elif driver == "3v3_roundrobin_lane_heroes":
        return camp_iterator_3v3_roundrobin_lane_heroes(driver_config)
    elif driver == "5v5dld_roundrobin_camp_heroes":
        return camp_iterator_5v5dld_roundrobin_camp_heroes(driver_config)
    elif driver == "3v3_roundrobin_camp_heroes":
        return camp_iterator_3v3_roundrobin_camp_heroes(driver_config)
    else:
        raise Exception("Unknown camp driver: %s" % driver)


if __name__ == "__main__":
    print("test default driver")
    camp_iter = camp_iterator()
    for i in range(10):
        print(next(camp_iter))

    print("test cycle_camps (with 2 scheme of camps)")
    camps = {
        "mode": "1v1",
        "camps": [
            [
                [{"hero_id": "A"}],
                [{"hero_id": "B"}],
            ],
            [
                [{"hero_id": "B"}],
                [{"hero_id": "A"}],
            ],
        ],
    }
    camp_iter = camp_iterator("cycle_camps", camps)
    for i in range(8):
        print(next(camp_iter))

    print("test camp_iterator_1v1_roundrobin_camp_heroes (with 3 heros)")
    camp_iter = camp_iterator("1v1_roundrobin_camp_heroes", ["A", "B", "C"])
    for i in range(18):
        print(next(camp_iter))

    print("test default driver 3v3")
    camp_iter = camp_iterator(default_mode=GameMode.G3v3)
    for i in range(10):
        print(next(camp_iter))

    assert GameMode.G3v3 == "3v3"
    assert GameMode.G1v1 == "1v1"
    assert GameMode.G5v5dld == "5v5dld"
    print("test camp_iterator_3v3_roundrobin_lane_heroes (with 2 heros each lane)")
    camp_iter = camp_iterator(
        "3v3_roundrobin_lane_heroes", [[190, 141], [173, 111], [107, 117]]
    )
    for i in range(10):
        print(next(camp_iter))

    print("test camp_iterator_5v5dld_roundrobin_camp_heroes")
    camp_iter = camp_iterator(
        "5v5dld_roundrobin_camp_heroes", [[190, 141, 173, 111, 107]]
    )
    for i in range(10):
        print(next(camp_iter))
