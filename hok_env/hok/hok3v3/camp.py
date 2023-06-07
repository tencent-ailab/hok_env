import itertools
import random

import os

# 8*8
def camp_iterator_six_hero_fixed_location():
    heros = [
        [{"name": "诸葛亮", "value": "190"}, {"name": "貂蝉", "value": "141"}],
        [{"name": "李元芳", "value": "173"}, {"name": "孙尚香", "value": "111"}],
        [{"name": "赵云", "value": "107"}, {"name": "钟无艳", "value": "117"}],
    ]
    camp_heros = [
        [x["value"] for x in hero_list] for hero_list in itertools.product(*heros)
    ]
    camps = [x for x in itertools.product(camp_heros, camp_heros)]
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


def camp_iterator_hero_cid_from_env():
    hero_cid_str = os.getenv("HERO_CID")
    hero_cid = hero_cid_str.split(",")
    while True:
        yield hero_cid[: len(hero_cid) // 2], hero_cid[len(hero_cid) // 2 :]


# 20 镜像对战
def camp_iterator_six_hero_mirror():
    heros = [
        [{"name": "诸葛亮", "value": "190"}, {"name": "貂蝉", "value": "141"}],
        [{"name": "李元芳", "value": "173"}, {"name": "孙尚香", "value": "111"}],
        [{"name": "赵云", "value": "107"}, {"name": "钟无艳", "value": "117"}],
    ]
    camp_heros = [x["value"] for hero_list in heros for x in hero_list]
    camps = [x for x in itertools.combinations(camp_heros, 3)]
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield (camp, camp)


def camp_iterator():
    driver = os.getenv("CAMP_DRIVER", "six_hero_fixed_location")
    if driver == "six_hero_mirror":
        return camp_iterator_six_hero_mirror()
    elif driver == "six_hero_fixed_location":
        return camp_iterator_six_hero_fixed_location()
    elif driver == "hero_cid_from_env":
        return camp_iterator_hero_cid_from_env()
    else:
        raise Exception("Unknown camp driver: %s" % driver)


if __name__ == "__main__":
    camp_iter = camp_iterator()
    for camp in enumerate(next(camp_iter)):
        print(camp)
