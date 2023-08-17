import os
from absl import app as absl_app
from absl import flags

work_dir = os.path.dirname(os.path.abspath(__file__))

from hok.hok3v3.server import BattleServer
from hok.hok3v3.lib import lib3v3 as interface
from hok.hok3v3.hero_config import interface_default_config

from agent.agent import Agent
from config.model_config import ModelConfig

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

flags.DEFINE_string("server_addr", "tcp://0.0.0.0:35400", "address of server")

DEFAULT_MODEL_PATH = os.path.join(work_dir, "model", "init")
flags.DEFINE_string("model_path", DEFAULT_MODEL_PATH, "path to checkpoint")
flags.DEFINE_string(
    "config_path", interface_default_config, "config file for interface"
)


def server(_):
    setup_logger()
    FLAGS = flags.FLAGS

    agent = Agent(
        Model(ModelConfig),
        None,
        backend=ModelConfig.backend,
    )
    agent.reset(model_path=FLAGS.model_path)

    lib_processor = interface.Interface()
    lib_processor.Init(FLAGS.config_path)
    lib_processor.SetEvalMode(True)

    server = BattleServer(agent, FLAGS.server_addr, lib_processor)
    server.run()


if __name__ == "__main__":
    absl_app.run(server)
