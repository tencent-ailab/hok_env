import os
import json

default_hero_config_file = os.path.join(
    os.path.dirname(__file__), "default_hero_config.json"
)
if os.getenv("KAIWU_V83_GAMECORE") == "1":
    interface_default_config = os.path.join(os.path.dirname(__file__), "config.v83.dat")
else:
    interface_default_config = os.path.join(os.path.dirname(__file__), "config.dat")


def get_default_hero_config():
    default_hero_config = {}
    with open(default_hero_config_file) as f:
        data = json.load(f)
        for _hero_config in data:
            default_hero_config[_hero_config["hero_id"]] = _hero_config
    return default_hero_config
