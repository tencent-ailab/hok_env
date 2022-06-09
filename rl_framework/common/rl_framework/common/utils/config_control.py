# -*- coding:utf-8 -*-

import os
import rapidjson
import configparser
from rl_framework.common.utils.common_func import Singleton


@Singleton
class ConfigControl(object):
    def __init__(self) -> None:
        self.config_file = None

    def parse_configue(self):
        if not self.config_file:
            return

        config = configparser.ConfigParser()
        config.read(self.config_file)

        # main conf
        self.run_mode = config.getint("main", "run_mode")
        self.log_dir = config.get("main", "log_dir")

        # aisrv conf
        self.max_tcp_count = config.getint("aisrv", "max_tcp_count")
        self.ip_address = config.get("aisrv", "ip_address")
        self.server_port = config.get("aisrv", "server_port")

        # actor conf

        # learner conf

    def set_configue_file(self, config_file):
        self.config_file = config_file


CONFIG = ConfigControl()
