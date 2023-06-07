import random
import os
import json
import itertools

HERO_DICT = {
    "luban": 112,
    "miyue": 121,
    "lvbu": 123,
    "libai": 131,
    "makeboluo": 132,
    "direnjie": 133,
    "guanyu": 140,
    "diaochan": 141,
    "luna": 146,
    "hanxin": 150,
    "huamulan": 154,
    "buzhihuowu": 157,
    "jvyoujing": 163,
    "houyi": 169,
    "zhongkui": 175,
    "ganjiangmoye": 182,
    "kai": 193,
    "gongsunli": 199,
    "peiqinhu": 502,
    "shangguanwaner": 513,
}


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
def camp_iterator_cycle_camps(camps=None):
    if camps is None:
        camps = {
            "mode": "1v1",
            "camps": [
                [
                    [{"hero_id": 112, "skill_id": 80115, "symbol": [1512, 1512]}],
                    [{"hero_id": 121, "skill_id": 80115, "symbol": [1512, 1512]}],
                ]
            ],
        }
    mode = camps["mode"]
    camps = camps["camps"]
    return _camp_iterator_shuffle_cycle(mode, camps)


# 多英雄阵容两两组合循环
def camp_iterator_roundrobin_camp_heroes(camp_heroes=None):
    if camp_heroes is None:
        camp_heroes = {
            "mode": "1v1",
            "camp_heroes": [
                [{"hero_id": 112, "skill_id": 80115, "symbol": [1512, 1512]}],
                [{"hero_id": 121, "skill_id": 80115, "symbol": [1512, 1512]}],
            ],
        }
    camps = [
        list(x)
        for x in itertools.product(
            camp_heroes["camp_heroes"], camp_heroes["camp_heroes"]
        )
    ]
    mode = camp_heroes["mode"]
    return _camp_iterator_shuffle_cycle(mode, camps)


def camp_iterator_1v1_roundrobin_camp_heroes(hero_ids):
    camp_heroes = {
        "mode": "1v1",
        "camp_heroes": [[{"hero_id": hero_id}] for hero_id in hero_ids],
    }
    return camp_iterator_roundrobin_camp_heroes(camp_heroes)


# 创建阵容配置的迭代器用于创建对战
# 当两个参数为空的时候, 从环境变量获取对应的参数
# 当相关环境变量未设置时使用默认值
def camp_iterator(driver=None, driver_config=None):
    if driver is None:
        driver = os.getenv("CAMP_DRIVER", "roundrobin_camp_heroes")

    if driver_config is None:
        driver_config_str = os.getenv(
            "CAMP_DRIVER_CONFIG",
            '{"mode": "1v1", "camp_heroes": [[{"hero_id": 121}], [{"hero_id": 112}]]}',
        )
        driver_config = json.loads(driver_config_str)

    if driver == "roundrobin_camp_heroes":
        return camp_iterator_roundrobin_camp_heroes(driver_config)
    elif driver == "cycle_camps":
        return camp_iterator_cycle_camps(driver_config)
    else:
        raise Exception("Unknown camp driver: %s" % driver)


if __name__ == "__main__":
    print("test default driver")
    camp_iter = camp_iterator()
    for i in range(10):
        print(next(camp_iter))

    print("test cycle_camps")
    camps = {
        "mode": "1v1",
        "camps": [
            [
                [{"hero_id": 190}],
                [{"hero_id": 105}],
            ],
            [
                [{"hero_id": 105}],
                [{"hero_id": 190}],
            ],
        ],
    }
    camp_iter = camp_iterator("cycle_camps", camps)
    for i in range(10):
        print(next(camp_iter))

    print("test camp_iterator_1v1_roundrobin_camp_heroes")
    camp_iter = camp_iterator_1v1_roundrobin_camp_heroes(HERO_DICT.values())
    for i in range(10):
        print(next(camp_iter))
