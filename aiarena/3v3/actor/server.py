import os

from absl import app as absl_app
from absl import flags

from hok.hok3v3.hero_config import interface_default_config
from hok.hok3v3.lib import lib3v3 as interface
from hok.hok3v3.server import BattleServer
from hok.common.camp import HERO_DICT

from rl_framework.common.logging import setup_logger
from rl_framework.common.logging import logger as LOG

from agent.agent import Agent
from config.model_config import ModelConfig
from config.config import Config
from kaiwu import kaiwu_info_example


def get_model_class(backend):
    if backend == "tensorflow":
        from model.tensorflow.model import Model
    elif backend == "pytorch":
        from model.pytorch.model import Model
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    else:
        raise NotImplementedError("backend=['tensorflow', 'pytorch']")
    return Model


work_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(work_dir, "model", "init")

flags.DEFINE_string("server_addr", "tcp://0.0.0.0:35400", "address of server")
flags.DEFINE_string("model_path", DEFAULT_MODEL_PATH, "path to checkpoint")
flags.DEFINE_string(
    "config_path", interface_default_config, "config file for interface"
)


def server(_):
    kaiwu_info_example()

    # 未设置默认为随机, -1表示随机选择一套装备, 0表示index为0的装备列表, 以此类推
    equip_config = {
        "houyi": 0,
        "yangyuhuan": 0,
        "caocao": 0,
        "jvyoujing": 0,
        "luban": 0,
        "bianque": 0,
        "buzhihuowu": 0,
        "yuji": 0,
        "sunwukong": 0,
    }

    model_config, config = ModelConfig, Config

    setup_logger(filename=None, level="INFO")
    Model = get_model_class(config.backend)

    FLAGS = flags.FLAGS
    agent = Agent(
        Model(model_config),
        None,
        backend=config.backend,
    )
    agent.reset(model_path=FLAGS.model_path)

    lib_processor = interface.Interface()
    lib_processor.Init(FLAGS.config_path)
    lib_processor.SetEvalMode(True)

    for hero_name, equip_index in equip_config.items():
        lib_processor.SetHeroEquipIndex(HERO_DICT.get(hero_name, 0), equip_index)
    LOG.info("equip_config: {}", lib_processor.m_hero_equip_idx)

    server = BattleServer(agent, FLAGS.server_addr, lib_processor)
    server.run()


if __name__ == "__main__":
    absl_app.run(server)
