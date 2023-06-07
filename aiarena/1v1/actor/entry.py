import sys

# sys.path.append(".")
sys.path.append("/aiarena/code/common/")
import random
import os
import logging

from absl import app as absl_app
from absl import flags


# TODO: 必须在tensorflow之前import influxdb?
from rl_framework.monitor import InfluxdbMonitorHandler
from actor import Actor
from agent import Agent as Agent
from sample_manager import SampleManager
from model import Model
from common_config import Config
from common_log import CommonLogger


from hok.hok1v1.camp import camp_iterator
import hok.hok1v1.lib.interface as interface
from hok.hok1v1.battle import Battle
from hok.hok1v1 import HoK1v1
from hok.hok1v1.env1v1 import interface_default_config

FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")

flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_boolean("single_test", 0, "test_mode")

flags.DEFINE_string(
    "config_path",
    os.getenv("INTERFACE_CONFIG_PATH", interface_default_config),
    "config file for interface",
)

flags.DEFINE_integer(
    "gamecore_req_timeout",
    30000,
    "millisecond timeout for gamecore to wait reply from server",
)

flags.DEFINE_string(
    "gc_server_addr",
    os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1"),
    "the gamecore server addr",
)

flags.DEFINE_string(
    "aiserver_ip",
    os.getenv("AI_SERVER_ADDR", "127.0.0.1"),
    "the actor ip",
)

flags.DEFINE_integer("max_episode", -1, "max number for run episode")
flags.DEFINE_string("monitor_server_addr", "127.0.0.1:8086", "monitor server addr")

MAP_SIZE = 100
AGENT_NUM = 2


#  gamecore as lib
def gc_as_lib(argv):
    Config.BATCH_SIZE = 1

    actor_id = FLAGS.actor_id
    agents = []
    game_id_init = "None"
    main_agent = random.randint(0, 1)

    load_models = []
    for m in FLAGS.agent_models.split(","):
        if len(m) > 0:
            load_models.append(m)
    if FLAGS.single_test:
        Config.SINGLE_TEST = True
    print(load_models)
    for i, m in enumerate(load_models):
        if m == "common_ai":
            load_models[i] = None

    print("load config.dat: ", FLAGS.config_path)
    lib_processor = interface.Interface()
    lib_processor.Init(FLAGS.config_path)

    game_launcher = Battle(
        server_addr=FLAGS.gc_server_addr,
        gamecore_req_timeout=FLAGS.gamecore_req_timeout,
    )

    addrs = []
    for i in range(AGENT_NUM):
        addrs.append("tcp://0.0.0.0:{}".format(35300 + actor_id * AGENT_NUM + i))

    env = HoK1v1(
        "actor-1v1-{}".format(actor_id),
        game_launcher,
        lib_processor,
        addrs,
        aiserver_ip=FLAGS.aiserver_ip,
    )

    for i in range(AGENT_NUM):
        agents.append(
            Agent(
                Model,
                FLAGS.model_pool_addr.split(";"),
                keep_latest=(i == main_agent),
            )
        )

    sample_manager = SampleManager(
        mem_pool_addr=FLAGS.mem_pool_addr,
        mem_pool_type="zmq",
        num_agents=AGENT_NUM,
        game_id=game_id_init,
    )

    gpu_ip = FLAGS.mem_pool_addr.split(":")[0]
    monitor_logger = logging.getLogger("monitor")
    monitor_logger.setLevel(logging.INFO)
    monitor_handler = InfluxdbMonitorHandler(gpu_ip)
    monitor_handler.setLevel(logging.INFO)
    monitor_logger.addHandler(monitor_handler)

    CommonLogger.set_config(actor_id)

    camp_iter = camp_iterator()

    actor = Actor(
        id=actor_id,
        agents=agents,
        monitor_logger=monitor_logger,
        camp_iter=camp_iter,
        max_episode=FLAGS.max_episode,
    )
    actor.set_sample_manager(sample_manager)
    actor.set_env(env)
    actor.run(load_models=load_models)


if __name__ == "__main__":
    absl_app.run(gc_as_lib)
