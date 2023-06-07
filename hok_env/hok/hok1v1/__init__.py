#!/usr/bin/env python
import os
import sys

# sys.path.append('.')
hok_path = os.getenv("HOK_GAMECORE_PATH")
cur_path = os.path.dirname(__file__)
sys.path.append(cur_path + "/proto_king/")

# sys.path.append(cur_path + '/lib/')
from hok.hok1v1.env1v1 import HoK1v1
