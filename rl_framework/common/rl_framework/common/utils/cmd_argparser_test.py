# -*- coding:utf-8 -*-

import sys
import unittest
from unittest.mock import patch
from rl_framework.common.utils.cmd_argparser import cmd_args_parse


class CmdArgParserTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_aisrv_args(self):
        testargs = ["proc", "--actor_adress", "0.0.0.0:8000"]

        with patch.object(sys, "argv", testargs):
            args = cmd_args_parse("aisrv")
            print(args.actor_adress)


if __name__ == "__main__":
    unittest.main()
