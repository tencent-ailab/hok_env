#!/usr/bin/env python
import os
import sys

# sys.path.append('.')
hok_path = os.getenv("HOK_GAMECORE_PATH")
cur_path = os.path.dirname(__file__)
sys.path.append(cur_path + "/proto_king/")

# sys.path.append(cur_path + '/lib/')
from hok.env1v1 import HoK1v1

# from hok.env1v1 import load_game

# Public API:
__all__ = ["HoK1v1", "GameRender", "__version__", "get_version", "GAMECORE_VERSION"]
