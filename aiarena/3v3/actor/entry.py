import random
import signal
import os
import sys
import logging
from absl import app as absl_app
from absl import flags

from config.model_config import ModelConfig
from config.config import Config
from actor import Actor
from agent.agent import Agent
from sample_manager import SampleManager as SampleManager

from hok.hok3v3.lib import lib3v3 as interface
from hok.hok3v3.env import Environment
from hok.hok3v3.server import AIServer
from hok.hok3v3.hero_config import get_default_hero_config, interface_default_config
from hok.common.gamecore_client import GamecoreClient, SimulatorType
from hok.common.camp import GameMode, camp_iterator
import rl_framework.common.logging as LOG

from rl_framework.monitor import InfluxdbMonitorHandler
from rl_framework.common.logging import setup_logger


if ModelConfig.backend == "tensorflow":
    from model.tensorflow.model import Model
elif ModelConfig.backend == "pytorch":
    from model.pytorch.model import Model
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
else:
    raise NotImplementedError("check ModelConfig, backend=['tensorflow', 'pytorch']")

FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")
flags.DEFINE_string(
    "gc_server_addr",
    os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1:23432"),
    "address of gamecore server",
)
flags.DEFINE_string(
    "aiserver_ip", os.getenv("AI_SERVER_ADDR", "127.0.0.1"), "the actor ip"
)

flags.DEFINE_integer("max_episode", -1, "max number for run episode")
flags.DEFINE_string("monitor_server_addr", "127.0.0.1:8086", "monitor server addr")
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
    "runtime_id_prefix",
    os.getenv("ACTOR_RUNTIME_ID_PREFIX", "actor-3v3"),
    "must not contain '_'",
)
flags.DEFINE_integer("port_begin", int(os.getenv("ACTOR_PORT_BEGIN", "35350")), "")
flags.DEFINE_integer("max_frame_num", int(os.getenv("MAX_FRAME_NUM", "20000")), "")
flags.DEFINE_boolean("debug_log", False, "use debug log level")

AGENT_NUM = 2


# gamecore as lib
def gc_as_lib(_):
    actor_id = FLAGS.actor_id
    if FLAGS.debug_log:
        setup_logger(
            filename="/aiarena/logs/actor/actor_{}.log".format(actor_id), level="DEBUG"
        )
    else:
        setup_logger(filename="/aiarena/logs/actor/actor_{}.log".format(actor_id))

    agents = []
    main_agent = random.randint(0, AGENT_NUM - 1)

    LOG.info("load config.dat: {}", FLAGS.config_path)
    lib_processor = interface.Interface()
    lib_processor.Init(FLAGS.config_path)

    agents = []
    aiservers = []
    for i in range(AGENT_NUM):
        agents.append(
            Agent(
                Model(ModelConfig),
                FLAGS.model_pool_addr.split(";"),
                keep_latest=(i == main_agent),
                backend=ModelConfig.backend,
                single_test=FLAGS.single_test,
            )
        )
        addr = f"tcp://0.0.0.0:{FLAGS.port_begin + actor_id * AGENT_NUM + i}"
        aiservers.append(AIServer(addr, lib_processor))

    if FLAGS.single_test:
        Config.SEND_SAMPLE_FRAME = 10
    game_launcher = GamecoreClient(
        server_addr=FLAGS.gc_server_addr,
        gamecore_req_timeout=FLAGS.gamecore_req_timeout,
        default_hero_config=get_default_hero_config(),
        max_frame_num=FLAGS.max_frame_num,
        simulator_type=SimulatorType.RemoteRepeat,
    )
    runtime_id = f"{FLAGS.runtime_id_prefix.replace('_', '-')}-{actor_id}"

    def signal_handler(signum, frame):
        game_launcher.stop_game(runtime_id)
        sys.exit(-1)

    signal.signal(signal.SIGTERM, signal_handler)
    env = Environment(
        aiservers,
        lib_processor,
        game_launcher,
        runtime_id=runtime_id,
    )

    sample_manager = SampleManager(
        mem_pool_addr=FLAGS.mem_pool_addr,
        num_agents=AGENT_NUM,
        single_test=FLAGS.single_test,
    )

    monitor_server_ip, monitor_server_port = FLAGS.monitor_server_addr.split(":")
    monitor_logger = logging.getLogger("monitor")
    monitor_logger.setLevel(logging.INFO)
    monitor_handler = InfluxdbMonitorHandler(
        ip=monitor_server_ip, port=monitor_server_port
    )
    monitor_handler.setLevel(logging.INFO)
    monitor_logger.addHandler(monitor_handler)

    camp_iter = camp_iterator(default_mode=GameMode.G3v3)

    actor = Actor(
        id=actor_id,
        agents=agents,
        env=env,
        sample_manager=sample_manager,
        camp_iter=camp_iter,
        max_episode=FLAGS.max_episode,
        monitor_logger=monitor_logger,
        send_sample_frame=Config.SEND_SAMPLE_FRAME,
    )
    try:
        actor.run()
    finally:
        game_launcher.stop_game(runtime_id)


if __name__ == "__main__":
    absl_app.run(gc_as_lib)
