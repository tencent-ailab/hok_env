from absl import app as absl_app
from absl import flags

from hok.hok3v3.server import AIServer

from agent.agent import Agent
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

flags.DEFINE_string("server_addr", "tcp://0.0.0.0:35400", "address of server")
flags.DEFINE_string("model_path", "./model/init", "path to checkpoint")


def server(_):
    FLAGS = flags.FLAGS

    agent = Agent(
        Model(ModelConfig),
        None,
        local_mode=True,
        backend=ModelConfig.backend,
    )
    agent.reset(model_path=FLAGS.model_path)

    server = AIServer(agent, FLAGS.server_addr)
    server.run()


if __name__ == "__main__":
    absl_app.run(server)
