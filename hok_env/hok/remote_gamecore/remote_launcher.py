import fcntl
import json
import logging
import os
import pickle as pkl
import sys
import time
import traceback

from hok.gamecore_utils import GameLauncher, ConfigModifier

MAX_PORT = 128


class TokenManager:
    def __init__(self, gamecore_path="~/.hok", expired_time=30 * 60):
        self.gamecore_path = gamecore_path
        self.expired_time = expired_time
        self.token_file = None

        logger = logging.get_logger("token")
        logger.setLevel(level=logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )

        file_handler = logging.FileHandler(
            os.path.join(self.gamecore_path, "token.log")
        )
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger

    def _load_token(self, modify=False):
        fname = os.path.join(self.gamecore_path, "token_table")

        if modify:
            try:
                self.token_file = open(fname, "rb+")
            except FileNotFoundError:
                self.token_file = open(fname, "wb+")
            fcntl.flock(self.token_file.fileno(), fcntl.LOCK_EX)
        else:
            self.token_file = open(fname, "rb")
            fcntl.flock(self.token_file.fileno(), fcntl.LOCK_SH)
        try:
            tlist = pkl.load(self.token_file)
        except Exception:  # pylint: disable=broad-except
            tlist = []
        return tlist

    def _dump_token(self, tlist, update=False):
        if update:
            self.token_file.truncate(0)
            self.token_file.seek(0, 0)
            pkl.dump(tlist, self.token_file)
        self.token_file.close()
        self.token_file = None

    def _remove_token(self, token):
        user_path = os.path.join(self.gamecore_path, "runtime/{}/".format(token))
        kill_command = (
            "ps -eo pgid,pid,command |grep sgame_simulator_ |"
            " grep \"cd {}\" |grep -v grep |awk '{{print -$1}}' |xargs kill -9".format(
                user_path
            )
        )
        os.system(kill_command)
        os.system("rm -rf {}".format(user_path))

    def _check_expired(self, token_info):
        now = time.time()
        if (now - token_info[1]) > self.expired_time:
            return True
        return False

    def _load_token_list(
        self,
    ):
        with open(os.path.join(self.gamecore_path, "token_list"), "r") as f:
            token_list = f.read().split("\n")
        self.token_list = []
        for t in token_list:
            t = t.strip()
            if len(t) > 0:
                self.token_list.append(t)

    def check_token(self, token, update=False):
        tlist = self._load_token(update)
        # print("="*10)
        # print(tlist)
        # print("=" * 10)
        new_tlist = []
        token_idx = -1
        for i, t in enumerate(tlist):
            if token == t[0]:
                token_idx = i
            if update:
                if self._check_expired(t) and t[0] != token:
                    self.logger.info(
                        "token {} expired, time {:.3f}min / {}min".format(
                            t[0], (time.time() - t[1]) / 60, self.expired_time / 60
                        )
                    )

                    self._remove_token(t[0])
                else:
                    print(
                        "token {} not expired, time {:.3f}min / {}min".format(
                            t[0], (time.time() - t[1]) / 60, self.expired_time / 60
                        )
                    )

                    new_tlist.append(t)
        if update:
            if token_idx == -1:
                # print("register new token", token)
                self.logger.info("register new token {}".format(token))
                new_tlist.append([token, time.time()])
            else:
                # print("renewal token", token)
                t = tlist[token_idx]
                self.logger.info(
                    "renewal {}, time {:.3f}min / {}min".format(
                        t[0], (time.time() - t[1]) / 60, self.expired_time / 60
                    )
                )

                tlist[token_idx][1] = time.time()
            self._dump_token(new_tlist, update=True)
            # print("=" * 10)
            # print(new_tlist)

            self.logger.info(
                "update token table from {} to {}".format(len(tlist), len(new_tlist))
            )

            return True

        self._dump_token(new_tlist, update=False)
        return token_idx > -1


def check_token(token, gamecore_path, update=False):
    return TokenManager(gamecore_path=gamecore_path).check_token(token, update)


class GameLauncherScript(GameLauncher):
    """
    This version not keep gamecore process alive with itself.

    """

    def __del__(self):
        pass

    def _load_token_list(
        self,
    ):
        with open(os.path.join(self.gamecore_path, "token_list"), "r") as f:
            token_list = f.read().split("\n")
        self.token_list = []
        for t in token_list:
            t = t.strip()
            if len(t) > 0:
                self.token_list.append(t)
        # self.token_list = [t.strip() for t in self.token_list]

    def _check_token(self, token, update):
        # self._load_token_list()
        self.user_token = token
        # available = token in self.token_list
        available = check_token(token, self.gamecore_path, update)
        assert available, "unavailable token {}".format(token)

    def _start_init(self, runtime_id):

        # running path / log path
        self.num_player = 2
        self.runtime_id = runtime_id

        self.user_path = os.path.join(
            self.gamecore_path, "runtime/{}".format(self.user_token)
        )
        self.runtime_path = os.path.join(
            self.gamecore_path, "runtime/{}/{}".format(self.user_token, self.runtime_id)
        )

        self.gc_proc = None
        self.proc_timeout = 1

        if not os.path.exists(self.user_path):
            os.makedirs(self.user_path, exist_ok=True)

        logger = logging.get_logger("user")
        logger.setLevel(level=logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )
        remote_log = os.path.join(self.gamecore_path, "remote_log")
        os.makedirs(remote_log, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(remote_log, "{}.log".format(self.user_token))
        )
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger

    def __init__(self, token, runtime_id, gamecore_path=None, update=None):
        # load token list and check
        if gamecore_path is None:
            gamecore_path = "~/.hok"
        # gamecore path
        gamecore_path = os.path.expanduser(gamecore_path)
        self.gamecore_path = gamecore_path
        self._check_token(token, update)
        self._start_init(runtime_id)

    def _kill_proc(self):
        # [!!!ATTENTION!!!] the kill command need be strict for killing specific process by runtime_id and token.
        kill_command = 'ps -eo pgid,command |grep sgame_simulator_ | grep "cd {};" |grep -v grep '.format(
            self.runtime_path
        )
        kill_list = os.popen(kill_command).readlines()
        kill_list = list(filter(lambda s: s and s.strip(), kill_list))

        if len(kill_list) > 0:
            print(
                "Process list to be killed[{}]: \n".format(len(kill_list)),
                "".join(kill_list),
            )
            kill_command = (
                "ps -eo pgid,pid,command |grep sgame_simulator_ | "
                "grep \"cd {};\" |grep -v grep |awk '{{print -$1}}' |xargs kill -9".format(
                    self.runtime_path
                )
            )
            print("Kill command: ", kill_command)
            os.system(kill_command)
        else:
            print("No running process to be killed.")

    def _kill_all(self):
        user_path = os.path.join(
            self.gamecore_path, "runtime/{}/".format(self.user_token)
        )
        kill_command = (
            "ps -eo pgid,pid,command |grep sgame_simulator_ |"
            " grep \"cd {}\" |grep -v grep |awk '{{print -$1}}' |xargs kill -9".format(
                user_path
            )
        )
        os.system(kill_command)

    def _close_proc(self):
        if hasattr(self, "gc_proc") and self.gc_proc is not None:
            del self.gc_proc
            self.gc_proc = None

    def _parse_config(self, config):
        runtime_id = config["id"]
        config_dict = config["config_dict"]
        common_config = config["common_config"]
        return runtime_id, config_dict, common_config

    def _get_all_files(self):
        try:
            file_list = os.listdir(self.runtime_path)
            ret = []
            for f in file_list:
                if f.split(".")[-1] in ["abs", "stat", "log"]:
                    if "_later" in f or "_detail" in f:
                        continue
                    ret.append(f)
        except Exception:  # pylint: disable=broad-except
            return []
        return ret

    def start(self, param):
        # print("start")

        config_dict, common_config = param["config_dict"], param["common_config"]
        runtime_id = param["runtime_id"]
        # conf, runtime_id, config_dict = self._parse_config(configs)

        assert os.path.exists(
            self.gamecore_path
        ), "ERROR: gamecore path {} not exists!".format(self.gamecore_path)
        assert 0 <= runtime_id < 128, "ERROR: illegal runtime_id!".format(runtime_id)

        try:
            os.mkdir(self.runtime_path)
        except FileExistsError:
            pass

        self._kill_proc()
        self.clear_runtime_path()

        common_config["init_abs"] = os.path.join(self.gamecore_path, "init_abs/1V1.abs")
        # print("start a new proc")

        self.config_modifier = ConfigModifier(
            gamecore_path=self.gamecore_path,
            runtime_id=runtime_id,
            runtime_path=self.runtime_path,
        )
        self.config_modifier.dump_config(config_dict, common_config)

        self._launch_proc(auto_kill=False)

    def kill(self, param):
        if param["runtime_id"] == -1:
            self._kill_all()
        else:
            self._kill_proc()

    def list_files(self, param):
        f_list = self._get_all_files()
        for f in f_list:
            print(f)

    def check_run(self, param):
        # os.system("ps -aux |grep sgame_simulator_ | grep \"{};\"| grep -v grep".format(self.runtime_path))
        check_command = (
            'ps -aux |grep sgame_simulator_ | grep "{};"| grep -v grep | wc -l'.format(
                self.runtime_path
            )
        )
        s = "".join(os.popen(check_command).readlines())
        if int(s) > 0:
            return 1
        else:
            return 0

    def get_path(self, param, path):
        f_list = self._get_all_files()
        if path in f_list:
            print(os.path.join(self.runtime_path, path))


if __name__ == "__main__":
    assert len(sys.argv) >= 4, "[ERROR] need type & token at least."
    # check token
    # print(sys.argv)
    # print("param", sys.argv[3])
    param = json.loads(sys.argv[3])
    runtime_id = param["runtime_id"]
    launcher = GameLauncherScript(
        sys.argv[2],
        runtime_id=runtime_id,
        update=(sys.argv[1] == "start"),
        gamecore_path="~/hok",
    )
    # print(launcher.runtime_path, launcher.gamecore_path)
    # parse config
    try:
        launcher.logger.info(" ".join(sys.argv[1:]))
        if sys.argv[1] == "start":
            launcher.start(
                param,
            )
        elif sys.argv[1] == "stop":
            launcher.kill(param)
            # launcher.kill_proc
        elif sys.argv[1] == "list":
            launcher.list_files(param)
        elif sys.argv[1] == "check":
            ret = launcher.check_run(param)
            exit(ret)
        elif sys.argv[1] == "get_path":
            launcher.get_path(param, sys.argv[4])
    except Exception as e:  # pylint: disable=broad-except
        traceback.print_exc()
        launcher.logger.error(str(e) + "\n" + traceback.format_exc())
    exit(0)
