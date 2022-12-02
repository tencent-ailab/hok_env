import sys

sys.path.append(".")
sys.path.append("/code/code/common/")
import random
import os

from absl import app as absl_app
from absl import flags


from actor import Actor

from agent import Agent as Agent
from algorithms.model.sample_manager import SampleManager as SampleManager
from algorithms.model.model import Model
from config.config import Config

FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_integer("max_step", 500, "max step of one round")
flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")
flags.DEFINE_string("gamecore_ip", "localhost", "address of gamecore")
flags.DEFINE_integer("thread_num", 1, "thread_num")

flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_integer("eval_number", -1, "battle number for evaluation")
flags.DEFINE_boolean("single_test", 0, "test_mode")

flags.DEFINE_string("gamecore_path", "~/.hok", "installation path of gamecore")
flags.DEFINE_string("game_log_path", "./game_log", "log path for game information")


MAP_SIZE = 100
AGENT_NUM = 2


#  gamecore as lib
def gc_as_lib(argv):
    # from gamecore.kinghonour.gamecore_client import GameCoreClient as Environment
    from hok import HoK1v1

    thread_id = 0
    actor_id = FLAGS.thread_num * FLAGS.actor_id + thread_id
    agents = []
    game_id_init = "None"
    main_agent = random.randint(0, 1)

    eval_number = FLAGS.eval_number
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
    eval_mode = eval_number > 0

    gc_server_addr = os.getenv("GAMECORE_SERVER_ADDR")
    ai_server_addr = os.getenv("AI_SERVER_ADDR")
    if (
        gc_server_addr is None
        or len(gc_server_addr) == 0
        or "127.0.0.1" in gc_server_addr
    ):
        # local gc server
        gc_server_addr = "127.0.0.1:23432"
        ai_server_addr = "127.0.0.1"
        remote_mode = 1
    else:
        # remote gc server
        remote_mode = 2
    remote_param = {
        "remote_mode": remote_mode,
        "gc_server_addr": gc_server_addr,
        "ai_server_addr": ai_server_addr,
    }
    gc_mode = os.getenv("GC_MODE")
    if gc_mode == "local":
        remote_param = None
    env = HoK1v1.load_game(
        runtime_id=actor_id,
        gamecore_path=FLAGS.gamecore_path,
        game_log_path=FLAGS.game_log_path,
        eval_mode=False,
        config_path="config.dat",
        remote_param=remote_param,
    )

    for i in range(AGENT_NUM):
        agents.append(
            Agent(
                Model,
                FLAGS.model_pool_addr.split(";"),
                keep_latest=(i == main_agent and not eval_mode),
                local_mode=eval_mode,
            )
        )

    sample_manager = SampleManager(
        mem_pool_addr=FLAGS.mem_pool_addr,
        mem_pool_type="zmq",
        num_agents=AGENT_NUM,
        game_id=game_id_init,
        local_mode=eval_mode,
    )
    actor = Actor(
        id=actor_id,
        agents=agents,
        gpu_ip=FLAGS.mem_pool_addr.split(":")[0]
    )
    actor.set_sample_managers(sample_manager)
    actor.set_env(env)
    actor.run(eval_mode=eval_mode, eval_number=eval_number, load_models=load_models)


if __name__ == "__main__":
    absl_app.run(gc_as_lib)
