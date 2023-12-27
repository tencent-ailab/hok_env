import os
import json

from rl_framework.common.logging import logger as LOG

work_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT_LIST_FILE = os.path.join(work_dir, "models.json")


def get_kaiwu_battle_info():
    """
    For competition battle info, parse following env
    CAMP_TYPE: RED or BLUE
    CAMP_BLUE_LINEUP_ID
    CAMP_BLUE_TEAM_ID
    CAMP_RED_LINEUP_ID
    CAMP_RED_TEAM_ID
    """

    ego_camp = os.getenv("CAMP_TYPE", "NOTSET").upper()

    enemy_camp = "NOTSET"
    if ego_camp == "BLUE":
        enemy_camp = "RED"
    elif ego_camp == "RED":
        enemy_camp = "BLUE"
    else:
        LOG.warning("Unknown camp info: {}", ego_camp)

    # 己方参数
    lineup_id = os.getenv("CAMP_" + ego_camp + "_LINEUP_ID", "-1")
    team_id = os.getenv("CAMP_" + ego_camp + "_TEAM_ID", "-1")

    # 敌方参数
    enemy_lineup_id = os.getenv("CAMP_" + enemy_camp + "_LINEUP_ID", "-1")
    enemy_team_id = os.getenv("CAMP_" + enemy_camp + "_TEAM_ID", "-1")
    return ego_camp, lineup_id, team_id, enemy_camp, enemy_lineup_id, enemy_team_id


def get_ckpt_list(ckpt_list_file):
    """
    parse ckpit list file injected by kaiwu, and return the dir list
    """
    if not os.path.exists(ckpt_list_file):
        LOG.info("{} not exists, ignore.", ckpt_list_file)
        return []

    with open(ckpt_list_file) as f:
        data = json.load(f)
    LOG.debug("load ckpt list file: {}", data)

    ckpt_dir_list = []
    for model_id in data.get("ids", []):
        ckpt_dir = os.path.join(work_dir, "model", f"model_{model_id}")
        if not os.path.exists(ckpt_dir):
            LOG.info("{} not exists, ignore.", ckpt_dir)
            continue

        LOG.info("List {}: {}", ckpt_dir, os.listdir(ckpt_dir))
        ckpt_dir_list.append(ckpt_dir)

    return ckpt_dir_list


def _kaiwu_info_example():
    (
        ego_camp,
        lineup_id,
        team_id,
        enemy_camp,
        enemy_lineup_id,
        enemy_team_id,
    ) = get_kaiwu_battle_info()
    LOG.info(
        "Get kaiwu battle info - camp:{} lineup_id:{} team_id:{}, enemy_camp:{}, enemy_lineup_id:{}, enemy_team_id:{}",
        ego_camp,
        lineup_id,
        team_id,
        enemy_camp,
        enemy_lineup_id,
        enemy_team_id,
    )

    ckpt_dir_list = get_ckpt_list(DEFAULT_CKPT_LIST_FILE)
    LOG.info("Get ckpt dir list: {}", ckpt_dir_list)


def kaiwu_info_example():
    try:
        _kaiwu_info_example()
    except:
        LOG.exception("kaiwu_info_example failed, ignore")


if __name__ == "__main__":
    kaiwu_info_example()
