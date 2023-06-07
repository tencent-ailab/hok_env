import sys

sys.path.append(".")
# sys.path.append('./lib')

import logging

from rl_framework.monitor import InfluxdbMonitorHandler
from rl_framework.common.logging import setup_logger

from absl import app as absl_app
from absl import flags
import random

from config.model_config import ModelConfig

if ModelConfig.backend == "tensorflow":
    from model.tensorflow.model import Model
elif ModelConfig.backend == "pytorch":
    from model.pytorch.model import Model
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
else:
    raise NotImplementedError("check ModelConfig, backend=['tensorflow', 'pytorch']")

from actor import Actor
from agent.agent import Agent
from sample_manager import SampleManager as SampleManager

FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_integer("max_step", 500, "max step of one round")
flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")
flags.DEFINE_string("gc_server_addr", "localhost:23432", "address of gamecore server")
flags.DEFINE_string("ai_server_ip", "localhost", "host of ai_server")
flags.DEFINE_integer("thread_num", 1, "thread_num")

flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_integer("eval_number", -1, "battle number for evaluation")
flags.DEFINE_integer("max_episode", -1, "max number for run episode")
flags.DEFINE_string("monitor_server_addr", "127.0.0.1:8086", "monitor server addr")
flags.DEFINE_boolean("single_test", 0, "test_mode")

AGENT_NUM = 2

# gamecore as lib
def gc_as_lib(argv):
    # TODO: used for different process
    from hok.hok3v3.gamecore_client import GameCoreClient as Environment

    thread_id = 0
    actor_id = FLAGS.thread_num * FLAGS.actor_id + thread_id
    setup_logger(filename="/aiarena/logs/actor/actor_{}.log".format(actor_id))
    agents = []
    game_id_init = "None"
    main_agent = random.randint(0, 1)

    eval_number = FLAGS.eval_number
    load_models = FLAGS.agent_models.split(",")
    for i, m in enumerate(load_models):
        if m == "common_ai":
            load_models[i] = None
    eval_mode = eval_number > 0

    for i in range(AGENT_NUM):
        agents.append(
            Agent(
                Model(ModelConfig),
                FLAGS.model_pool_addr.split(";"),
                keep_latest=(i == main_agent and not eval_mode),
                local_mode=eval_mode,
                backend=ModelConfig.backend,
                single_test=FLAGS.single_test,
            )
        )
    env = Environment(
        host=FLAGS.ai_server_ip, seed=actor_id, gc_server=FLAGS.gc_server_addr
    )
    sample_manager = SampleManager(
        mem_pool_addr=FLAGS.mem_pool_addr,
        mem_pool_type="zmq",
        num_agents=AGENT_NUM,
        game_id=game_id_init,
        local_mode=eval_mode,
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

    actor = Actor(
        id=actor_id,
        agents=agents,
        max_episode=FLAGS.max_episode,
        monitor_logger=monitor_logger,
    )
    actor.set_sample_managers(sample_manager)
    actor.set_env(env)
    actor.run(eval_mode=eval_mode, eval_number=eval_number, load_models=load_models)


if __name__ == "__main__":
    absl_app.run(gc_as_lib)
