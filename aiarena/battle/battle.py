import os
import signal
import sys
import threading
from typing import NamedTuple, List
from collections import defaultdict

from absl import app as absl_app
from absl import flags

from hok.common.gamecore_client import GamecoreClient
from hok.common.camp import camp_iterator, GameMode, thread_safe_iterator
from hok.common.server_process import ServerProcess
from hok.common.log import logger as LOG, setup_logger

from hok.hok1v1.hero_config import (
    get_default_hero_config as get_default_hero_config_1v1,
)

from hok.hok3v3.hero_config import (
    get_default_hero_config as get_default_hero_config_3v3,
)

from hok.hok5v5dld.hero_config import (
    get_default_hero_config as get_default_hero_config_5v5dld,
)


server_param = (
    "path to local dir if driver is local_dir, "
    + "path to tar dir if driver is local_tar, "
    + "url to download if driver is url, server ip if driver is server"
)

server_driver_param = (
    "local_dir(start_server from local dir), "
    + "local_tar(extract server for tar file and start), "
    + "url(download server from url, extract and start), "
    + "server(for started server), "
    + "common_ai (for common ai)"
)

for i in range(2):
    flags.DEFINE_string(f"server_{i}", f"server_{i}", f"server {i}: " + server_param)
    flags.DEFINE_string(f"path_{i}", f"server_{i}", f"server {i}")
    flags.DEFINE_integer(f"port_{i}", 35500 + i, f"port for server {i}")
    flags.DEFINE_string(
        f"logfile_{i}", f"/aiarena/logs/server_{i}.log", f"log for server {i}"
    )
    flags.DEFINE_string(
        f"driver_{i}", "local_dir", f"server driver {i}: " + server_driver_param
    )

flags.DEFINE_integer("wait_port_timeout", 30, "seconds wait for server")
flags.DEFINE_integer(
    "gamecore_req_timeout",
    30000,
    "millisecond timeout for gamecore to wait reply from server",
)
flags.DEFINE_integer(
    "max_episode", 1, "max episodes for each battle thread to run", short_name="n"
)
flags.DEFINE_boolean(
    "camp_swap",
    0,
    "swap camp for each camp config (max_episode should be manually doubled)",
)
flags.DEFINE_string("gamecore_server", "127.0.0.1:23432", "gamecore server address")
flags.DEFINE_string(
    "task_id",
    os.getenv("BATTLE_TASK_ID"),
    "task_id, gamecore_server will report gamestat if set",
)

GamecoreClientConfig = NamedTuple(
    "GamecoreClientConfig", [("server_addr", str), ("req_timeout", int)]
)

ServerProcessConfig = NamedTuple(
    "ServerProcessConfig",
    [("port", int), ("server", str), ("path", str), ("driver", str), ("logfile", str)],
)

flags.DEFINE_integer("concurrency", 1, "", short_name="c")

TaskConfig = NamedTuple(
    "TaskConfig",
    [
        ("wait_port_timeout", int),
        ("max_episode", int),
        ("task_id", str),
        ("camp_swap", bool),
    ],
)


class Battle:
    def __init__(
        self,
        gamecore: GamecoreClientConfig,
        servers: List[ServerProcessConfig],
        task: TaskConfig,
        camp_iter,
    ) -> None:
        self.gamecore = gamecore
        self.servers = servers
        self.task = task
        self.camp_iter = camp_iter
        self.client = None
        self.stop = False

    def get_swap_task_id(self, task_id):
        if task_id:
            return task_id + "_swap"
        return task_id

    def get_runtime_id(self, servers):
        return "r-{}-{}".format(servers[0].port, servers[1].port)

    def run(
        self,
    ):
        camp_hero_list = next(self.camp_iter)
        game_mode = camp_hero_list["mode"]

        default_hero_config = {}
        if game_mode == GameMode.G1v1:
            default_hero_config = get_default_hero_config_1v1()
        elif game_mode == GameMode.G3v3:
            default_hero_config = get_default_hero_config_3v3()
        elif game_mode == GameMode.G5v5dld:
            default_hero_config = get_default_hero_config_5v5dld()

        self.client = GamecoreClient(
            server_addr=self.gamecore.server_addr,
            gamecore_req_timeout=self.gamecore.req_timeout,
            default_hero_config=default_hero_config,
        )
        runtime_id = self.get_runtime_id(self.servers)
        self.client.stop_game(runtime_id)

        server_processes = [ServerProcess() for _ in range(len(self.servers))]
        for i, server in enumerate(server_processes):
            server.start(
                self.servers[i].server,
                self.servers[i].path,
                self.servers[i].port,
                self.servers[i].logfile,
                self.servers[i].driver,
            )

        for server in server_processes:
            server.wait_server_started(self.task.wait_port_timeout)

        episode_cnt = 0

        while episode_cnt < self.task.max_episode and not self.stop:
            server_config = [s.get_server_addr() for s in server_processes]
            task_id = self.task.task_id
            if self.task.camp_swap and episode_cnt % 2 == 1:
                server_config = server_config[::-1]
                task_id = self.get_swap_task_id(task_id)

            LOG.info(
                "Start game: episode_cnt({}) task_id({}) runtime_id({}) camp_hero_list({})",
                episode_cnt,
                task_id,
                runtime_id,
                camp_hero_list,
            )

            self.client.start_game(
                runtime_id,
                server_config,
                camp_hero_list,
                task_id=task_id,
                eval_mode=True,
            )
            self.client.wait_game(runtime_id)
            episode_cnt += 1

            if (
                self.task.camp_swap and episode_cnt % 2 == 0
            ) or not self.task.camp_swap:
                camp_hero_list = next(self.camp_iter)

        for server in server_processes:
            server.stop()

    def stop_game(self):
        if not self.client or self.stop:
            return

        self.stop = True
        runtime_id = self.get_runtime_id(self.servers)
        self.client.stop_game(runtime_id)

    def print_task_result(self):
        if not self.client:
            LOG.warning("gamecore client is None, skip")
            return

        if not self.task.task_id:
            LOG.info("task_id is '{}', skip getting result", self.task.task_id)
            return

        game_states = self.client.task_detail(self.task.task_id)
        game_states_swap = self.client.task_detail(
            self.get_swap_task_id(self.task.task_id)
        )

        win_num = defaultdict(int)
        total_num = 0

        for game_state in game_states:
            total_num += 1
            win_num[0] += game_state["Camp"][0]["Score"]
            win_num[1] += game_state["Camp"][1]["Score"]

        for game_state in game_states_swap:
            total_num += 1
            win_num[0] += game_state["Camp"][1]["Score"]
            win_num[1] += game_state["Camp"][0]["Score"]

        if total_num == 0:
            LOG.warning("No game state task_id({})", self.task.task_id)
        else:
            LOG.info(
                "Win Rate: TaskID({task_id}) Agent1({win_0}/{total_num} {win_rate_0:.2f}) Agent2({win_1}/{total_num} {win_rate_1:.2f})",
                task_id=self.task.task_id,
                win_0=win_num[0],
                total_num=total_num,
                win_rate_0=win_num[0] / total_num,
                win_1=win_num[1],
                win_rate_1=win_num[1] / total_num,
            )
            self.client.task_remove(self.task.task_id)
            self.client.task_remove(self.get_swap_task_id(self.task.task_id))


def main(_):
    setup_logger()
    # parse config from flags
    FLAGS = flags.FLAGS

    camp_iter = camp_iterator()
    camp_iter = thread_safe_iterator(camp_iter)

    gamecore_config = GamecoreClientConfig(
        server_addr=FLAGS.gamecore_server, req_timeout=FLAGS.gamecore_req_timeout
    )
    task_config = TaskConfig(
        wait_port_timeout=FLAGS.wait_port_timeout,
        max_episode=FLAGS.max_episode,
        task_id=FLAGS.task_id,
        camp_swap=FLAGS.camp_swap,
    )
    LOG.info("Gamecore config: {}", gamecore_config)
    LOG.info("Task config: {}", task_config)

    SERVER_NUM = 2
    threads = []
    battles = []
    for thread_idx in range(FLAGS.concurrency):
        servers_config = []
        for i in range(SERVER_NUM):
            server_config = ServerProcessConfig(
                server=FLAGS[f"server_{i}"].value,
                path=FLAGS[f"path_{i}"].value,
                port=FLAGS[f"port_{i}"].value + thread_idx * SERVER_NUM,
                logfile=FLAGS[f"logfile_{i}"].value,
                driver=FLAGS[f"driver_{i}"].value,
            )
            servers_config.append(server_config)

        LOG.info("Server config for thread {}: {}", thread_idx, servers_config)

        battle = Battle(
            gamecore=gamecore_config,
            servers=servers_config,
            task=task_config,
            camp_iter=camp_iter,
        )
        battles.append(battle)

        thread = threading.Thread(target=battle.run)

        thread.start()
        threads.append(thread)

    def signal_handler(signum, frame):
        LOG.info("Get signal, stop game: {} {}", signum, frame)
        for battle in battles:
            battle.stop_game()
        sys.exit(-1)

    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for thread in threads:
            thread.join()
    finally:
        for battle in battles:
            battle.stop_game()

        for battle in battles:
            # reuse gamecore_client in battle, and show the result of task
            battle.print_task_result()
            break


if __name__ == "__main__":
    absl_app.run(main)
