from absl import app as absl_app
from absl import flags

from hok.battle import Battle
from hok.camp import camp_iterator

flags.DEFINE_string(
    "server_0",
    "server_0",
    "server 0: path to local dir if driver is local_dir, path to tar dir if driver is local_tar, url to download if driver is url, server ip if driver is server ",
)

flags.DEFINE_string("server_path_0", "server_0", "server 0")
flags.DEFINE_integer("server_port_0", 35350, "port for server 0")
flags.DEFINE_string(
    "server_logfile_0", "/aiarena/logs/server_0.log", "log for server 0"
)
flags.DEFINE_string(
    "server_driver_0",
    "local_dir",
    "server driver 0: local_dir(start_server from local dir), local_tar(extract server for tar file and start), url(download server from url, extract and start), server(for started server), common_ai (for common ai)",
)

flags.DEFINE_string("server_1", "server_1", "server_1: same with server_0")
flags.DEFINE_string("server_path_1", "server_1", "server 1")
flags.DEFINE_integer("server_port_1", 35351, "port for server 1")
flags.DEFINE_string(
    "server_logfile_1", "/aiarena/logs/server_1.log", "log for server 1"
)
flags.DEFINE_string(
    "server_driver_1",
    "local_dir",
    "server driver 1: same with server_driver_0",
)

flags.DEFINE_integer("wait_port_timeout", 30, "seconds wait for server")
flags.DEFINE_integer(
    "gamecore_req_timeout",
    30000,
    "millisecond timeout for gamecore to wait reply from server",
)

flags.DEFINE_string("gamecore_server", "127.0.0.1:23432", "gamecore server address")


def server(_):
    FLAGS = flags.FLAGS

    b = Battle(
        server_addr=FLAGS.gamecore_server,
        gamecore_req_timeout=FLAGS.gamecore_req_timeout,
    )
    runtime_id = "r-{}-{}".format(FLAGS.server_port_0, FLAGS.server_port_1)
    b.stop_battle(runtime_id)

    servers = []

    servers.append(
        b.start_server(
            FLAGS.server_0,
            FLAGS.server_path_0,
            FLAGS.server_port_0,
            FLAGS.server_logfile_0,
            FLAGS.server_driver_0,
        )
    )

    servers.append(
        b.start_server(
            FLAGS.server_1,
            FLAGS.server_path_1,
            FLAGS.server_port_1,
            FLAGS.server_logfile_1,
            FLAGS.server_driver_1,
        )
    )

    b.wait_server(
        servers,
        FLAGS.wait_port_timeout,
    )

    camp_iter = camp_iterator()
    camp_hero_list = next(camp_iter)

    b.start_battle(runtime_id, servers, camp_hero_list)
    b.wait_battle(runtime_id)


if __name__ == "__main__":
    absl_app.run(server)
