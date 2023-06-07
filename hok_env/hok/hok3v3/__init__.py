#!/usr/bin/env python
import os
import sys

hok_path = os.getenv("HOK_GAMECORE_PATH")
cur_path = os.path.dirname(__file__)
sys.path.append(cur_path)
sys.path.append(cur_path + "/lib/")

CONFIG_DAT = os.path.join(cur_path, "config.dat")
