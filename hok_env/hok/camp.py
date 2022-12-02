import os


def camp_iterator_hero_cid_from_env():
    hero_cid_str = os.getenv("HERO_CID")
    hero_cid = hero_cid_str.split(",")
    while True:
        yield hero_cid[: len(hero_cid) // 2], hero_cid[len(hero_cid) // 2 :]


def camp_iterator():
    driver = os.getenv("CAMP_DRIVER", "six_hero_fixed_location")
    if driver == "hero_cid_from_env":
        return camp_iterator_hero_cid_from_env()
    else:
        raise Exception("Unknown camp driver: %s" % driver)


if __name__ == "__main__":
    camp_iter = camp_iterator()
    for camp in enumerate(next(camp_iter)):
        print(camp)
