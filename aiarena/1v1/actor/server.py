import sys

sys.path.append("/aiarena/code/common/")

from absl import app as absl_app
from absl import flags

from hok.hok1v1.server import AIServer

from agent import Agent
from model import Model

flags.DEFINE_string("server_addr", "tcp://0.0.0.0:35400", "address of server")
flags.DEFINE_string("model_path", "checkpoint", "path to checkpoint")


def server(_):
    FLAGS = flags.FLAGS

    agent = Agent(
        Model,
        model_pool_addr=None,
    )
    agent.reset(model_path=FLAGS.model_path)

    server = AIServer(agent, FLAGS.server_addr)
    server.run()


if __name__ == "__main__":
    absl_app.run(server)
