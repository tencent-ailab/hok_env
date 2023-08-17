import os

from absl import app as absl_app
from absl import flags

from hok.common.gamecore_client import GamecoreClient
from hok.common.camp import camp_iterator, GameMode
from hok.common.server_process import ServerProcess

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
flags.DEFINE_integer("max_episode", 1, "max episodes to run")
flags.DEFINE_boolean("camp_swap", 0, "swap camp for each camp config (max_episode should be manually doubled)")
flags.DEFINE_string("gamecore_server", "127.0.0.1:23432", "gamecore server address")
flags.DEFINE_string("task_id", os.getenv("BATTLE_TASK_ID"), "task_id, gamecore_server will report gamestat if set")


def server(_):
    camp_iter = camp_iterator()
    camp_hero_list = next(camp_iter)
    game_mode = camp_hero_list["mode"]

    default_hero_config = {}
    if game_mode == GameMode.G1v1:
        default_hero_config = get_default_hero_config_1v1()
    elif game_mode == GameMode.G3v3:
        default_hero_config = get_default_hero_config_3v3()
    elif game_mode == GameMode.G5v5dld:
        default_hero_config = get_default_hero_config_5v5dld()

    FLAGS = flags.FLAGS

    client = GamecoreClient(
        server_addr=FLAGS.gamecore_server,
        gamecore_req_timeout=FLAGS.gamecore_req_timeout,
        default_hero_config=default_hero_config,
    )
    runtime_id = "r-{}-{}".format(FLAGS.port_0, FLAGS.port_1)
    client.stop_game(runtime_id)

    servers = [ServerProcess() for _ in range(2)]
    for i, server in enumerate(servers):
        server.start(
            FLAGS[f"server_{i}"].value,
            FLAGS[f"path_{i}"].value,
            FLAGS[f"port_{i}"].value,
            FLAGS[f"logfile_{i}"].value,
            FLAGS[f"driver_{i}"].value,
        )

    for server in servers:
        server.wait_server_started(FLAGS.wait_port_timeout)

    episode_cnt = 0

    while episode_cnt < FLAGS.max_episode:
        server_config = [s.get_server_addr() for s in servers]
        task_id = FLAGS.task_id
        if FLAGS.camp_swap and episode_cnt % 2 == 1:
            server_config = server_config[::-1]
            task_id += "_swap"
        client.start_game(
            runtime_id,
            server_config,
            camp_hero_list,
            task_id=task_id,
            eval_mode=True,
        )
        client.wait_game(runtime_id)
        episode_cnt += 1

        if (FLAGS.camp_swap and episode_cnt % 2 == 0) or not FLAGS.camp_swap:
            camp_hero_list = next(camp_iter)

    for server in servers:
        server.stop()


if __name__ == "__main__":
    absl_app.run(server)
