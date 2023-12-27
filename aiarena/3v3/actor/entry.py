import logging
import os
import random
import signal
import sys

from absl import app as absl_app
from absl import flags
import psutil

from hok.common.camp import GameMode, camp_iterator
from hok.common.gamecore_client import GamecoreClient, SimulatorType
from hok.hok3v3.env import Environment
from hok.hok3v3.hero_config import get_default_hero_config, interface_default_config
from hok.hok3v3.lib import lib3v3 as interface
from hok.hok3v3.server import AIServer
from hok.hok3v3.reward import update_reward_config

from rl_framework.monitor import InfluxdbMonitorHandler
from rl_framework.common.logging import logger as LOG
from rl_framework.common.logging import setup_logger

from actor import Actor
from agent.agent import Agent
from sample_manager import SampleManager as SampleManager
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


def select_mempool(actor_id, actor_num, mempool_list):
    mempool_num = len(mempool_list)
    LOG.info("mempool list {}: {}", mempool_num, mempool_list)

    if actor_num % mempool_num and actor_id // mempool_num == actor_num // mempool_num:
        idx = random.randint(0, mempool_num - 1)
    else:
        idx = actor_id % mempool_num

    LOG.info("select mempool {}: {}", idx, mempool_list[idx])
    return mempool_list[idx]


def auto_bind_cpu(actor_id, actor_num):
    p = psutil.Process(os.getpid())
    cpu_ids = p.cpu_affinity() or []
    LOG.info("cpu_ids: {}", cpu_ids)
    if len(cpu_ids) == actor_num:
        cpu_id = cpu_ids[actor_id % len(cpu_ids)]
        p.cpu_affinity([cpu_id])
        LOG.info("bind actor_{} cpu_{}", actor_id, cpu_id)


AGENT_NUM = 2


def run(
    actor_id,
    config_path,
    model_pool_addr,
    single_test,
    port_begin,
    gc_server_addr,
    gamecore_req_timeout,
    max_frame_num,
    runtime_id_prefix,
    aiserver_ip,
    mem_pool_addr_list,
    max_episode,
    monitor_server_addr,
    config,
    model_config=None,
    log_file=None,
    debug_log=None,
):
    kaiwu_info_example()
    try:
        log_file = log_file or "/aiarena/logs/actor/actor_{}.log".format(actor_id)
        if debug_log:
            setup_logger(filename=log_file, level="DEBUG")
        else:
            setup_logger(log_file)

        _run(
            actor_id,
            config_path,
            model_pool_addr,
            single_test,
            port_begin,
            gc_server_addr,
            gamecore_req_timeout,
            max_frame_num,
            runtime_id_prefix,
            aiserver_ip,
            mem_pool_addr_list,
            max_episode,
            monitor_server_addr,
            config,
            model_config,
        )
    except SystemExit:
        LOG.error("Actor terminated")
        raise
    except Exception:
        LOG.exception("Actor failed.")
        raise


def _run(
    actor_id,
    config_path,
    model_pool_addr,
    single_test,
    port_begin,
    gc_server_addr,
    gamecore_req_timeout,
    max_frame_num,
    runtime_id_prefix,
    aiserver_ip,
    mem_pool_addr_list,
    max_episode,
    monitor_server_addr,
    config,
    model_config,
):
    if config.auto_bind_cpu:
        auto_bind_cpu(actor_id, config.actor_num)

    agents = []
    main_agent = random.randint(0, AGENT_NUM - 1)

    LOG.info("load config.dat: {}", config_path)
    lib_processor = interface.Interface()
    lib_processor.Init(config_path)
    update_reward_config(lib_processor, config.reward_config)

    agents = []
    aiservers = []
    Model = get_model_class(config.backend)

    for i in range(AGENT_NUM):
        agents.append(
            Agent(
                Model(model_config),
                model_pool_addr.split(";"),
                keep_latest=(i == main_agent),
                backend=config.backend,
                single_test=single_test,
            )
        )
        addr = f"tcp://0.0.0.0:{port_begin + actor_id * AGENT_NUM + i}"
        aiservers.append(AIServer(addr, lib_processor))

    game_launcher = GamecoreClient(
        server_addr=gc_server_addr,
        gamecore_req_timeout=gamecore_req_timeout,
        default_hero_config=get_default_hero_config(),
        max_frame_num=max_frame_num,
        simulator_type=SimulatorType.RemoteRepeat,
    )

    runtime_id = f"{runtime_id_prefix.replace('_', '-')}-{actor_id}"

    def signal_handler(signum, frame):
        game_launcher.stop_game(runtime_id)
        sys.exit(-1)

    signal.signal(signal.SIGTERM, signal_handler)

    env = Environment(
        aiservers,
        lib_processor,
        game_launcher,
        runtime_id=runtime_id,
        aiserver_ip=aiserver_ip,
    )

    mempool = select_mempool(actor_id, config.actor_num, mem_pool_addr_list)
    sample_manager = SampleManager(
        mem_pool_addr=mempool,
        num_agents=AGENT_NUM,
        single_test=single_test,
        sample_one_size=model_config.sample_one_size,
        lstm_unit_size=model_config.LSTM_UNIT_SIZE,
        lstm_time_steps=model_config.LSTM_TIME_STEPS,
        gamma=config.GAMMA,
        lamda=config.LAMDA,
    )

    monitor_ip, monitor_port = monitor_server_addr.split(":")
    monitor_logger = logging.getLogger("monitor")
    monitor_logger.setLevel(logging.INFO)
    monitor_handler = InfluxdbMonitorHandler(monitor_ip, monitor_port)
    monitor_handler.setLevel(logging.INFO)
    monitor_logger.addHandler(monitor_handler)

    camp_iter = camp_iterator(default_mode=GameMode.G3v3)

    SEND_SAMPLE_FRAME = 10 if single_test else config.SEND_SAMPLE_FRAME
    actor = Actor(
        id=actor_id,
        agents=agents,
        env=env,
        sample_manager=sample_manager,
        camp_iter=camp_iter,
        max_episode=max_episode,
        monitor_logger=monitor_logger,
        send_sample_frame=SEND_SAMPLE_FRAME,
    )
    try:
        actor.run()
    finally:
        game_launcher.stop_game(runtime_id)


def main(_):
    FLAGS = flags.FLAGS
    mem_pool_addr_list = FLAGS.mem_pool_addr.strip().split(";")
    monitor_ip = mem_pool_addr_list[0].split(":")[0]
    monitor_server_addr = f"{monitor_ip}:8086"
    from config.config import Config
    from config.model_config import ModelConfig

    run(
        FLAGS.actor_id,
        FLAGS.config_path,
        FLAGS.model_pool_addr,
        FLAGS.single_test,
        FLAGS.port_begin,
        FLAGS.gc_server_addr,
        FLAGS.gamecore_req_timeout,
        FLAGS.max_frame_num,
        FLAGS.runtime_id_prefix,
        FLAGS.aiserver_ip,
        mem_pool_addr_list,
        FLAGS.max_episode,
        monitor_server_addr,
        Config,
        ModelConfig,
        debug_log=FLAGS.debug_log,
    )


if __name__ == "__main__":
    flags.DEFINE_integer("actor_id", 0, "actor id")
    flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
    flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")

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
        os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1:23432"),
        "address of gamecore server",
    )
    flags.DEFINE_string(
        "aiserver_ip", os.getenv("AI_SERVER_ADDR", "127.0.0.1"), "the actor ip"
    )

    flags.DEFINE_integer("max_episode", -1, "max number for run episode")
    flags.DEFINE_string("monitor_server_addr", "127.0.0.1:8086", "monitor server addr")

    flags.DEFINE_string(
        "runtime_id_prefix",
        os.getenv("ACTOR_RUNTIME_ID_PREFIX", "actor-3v3"),
        "must not contain '_'",
    )
    flags.DEFINE_integer("port_begin", int(os.getenv("ACTOR_PORT_BEGIN", "35350")), "")
    flags.DEFINE_integer("max_frame_num", int(os.getenv("MAX_FRAME_NUM", "20000")), "")
    flags.DEFINE_boolean("debug_log", False, "use debug log level")
    absl_app.run(main)
