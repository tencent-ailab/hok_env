import sys
import os

work_dir = os.path.dirname(os.path.abspath(__file__))

# chdir to work_dir to access the config.json with relative path
os.chdir(work_dir)

# sys.path.append("/aiarena/code/") add common to path
sys.path.append(os.path.dirname(work_dir))

from absl import app as absl_app
from absl import flags

from hok.hok1v1.server import AIServer

from custom import Agent
from model import get_model_class
from common.config import Config
from rl_framework.common.logging import setup_logger

DEFAULT_MODEL_PATH = os.path.join(work_dir, "model", "init")

flags.DEFINE_string("server_addr", "tcp://0.0.0.0:35400", "address of server")
flags.DEFINE_string("model_path", DEFAULT_MODEL_PATH, "path to checkpoint")


def server(_):
    setup_logger(filename=None, level="INFO")

    FLAGS = flags.FLAGS
    Model = get_model_class(Config.backend)

    agent = Agent(
        Model(),
        model_pool_addr=None,
        config=Config,
    )
    agent.reset(model_path=FLAGS.model_path)

    server = AIServer(agent, FLAGS.server_addr)
    server.run()


if __name__ == "__main__":
    absl_app.run(server)
