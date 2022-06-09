# -*- coding:utf-8 -*-

import unittest
from rl_framework.common.utils.config_control import CONFIG


class ConfigControlTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_load_configure(self):
        configue_file = "/data/projects/rl_framework/conf/configue.json"

        CONFIG.set_configue_file(configue_file)
        CONFIG.parse_configue()

        print(CONFIG.ip_address)


if __name__ == "__main__":
    unittest.main()
