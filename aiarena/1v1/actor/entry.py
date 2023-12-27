import logging
import os
import os
import random
import signal
import sys

from absl import app as absl_app
from absl import flags
import psutil

from hok.common.camp import camp_iterator
from hok.common.gamecore_client import GamecoreClient, SimulatorType
from hok.hok1v1 import HoK1v1
from hok.hok1v1.env1v1 import interface_default_config
from hok.hok1v1.hero_config import get_default_hero_config
import hok.hok1v1.lib.interface as interface

# TODO: 必须在tensorflow之前import influxdb?
from rl_framework.monitor import InfluxdbMonitorHandler
from rl_framework.common.logging import logger as LOG
from rl_framework.common.logging import setup_logger

from actor import Actor
from custom import Agent
from sample_manager import SampleManager
from model import get_model_class

work_dir = os.path.dirname(os.path.abspath(__file__))

# sys.path.append("/aiarena/code/") add common to path
sys.path.append(os.path.dirname(work_dir))

AGENT_NUM = 2


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
):
    if config.auto_bind_cpu:
        auto_bind_cpu(actor_id, config.actor_num)

    # chdir to work_dir to access the config.json with relative path
    os.chdir(work_dir)

    agents = []
    game_id_init = "None"
    main_agent = random.randint(0, AGENT_NUM - 1)

    LOG.info("load config.dat: {}", config_path)
    lib_processor = interface.Interface()
    lib_processor.Init(config_path)
    Model = get_model_class(config.backend)

    for i in range(AGENT_NUM):
        agents.append(
            Agent(
                Model(),
                model_pool_addr.split(";"),
                config=config,
                keep_latest=(i == main_agent),
                single_test=single_test,
            )
        )

    addrs = []
    for i in range(AGENT_NUM):
        addrs.append(f"tcp://0.0.0.0:{port_begin + actor_id * AGENT_NUM + i}")

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

    env = HoK1v1(
        runtime_id,
        game_launcher,
        lib_processor,
        addrs,
        aiserver_ip=aiserver_ip,
    )

    mempool = select_mempool(actor_id, config.actor_num, mem_pool_addr_list)
    sample_manager = SampleManager(
        mem_pool_addr=mempool,
        mem_pool_type="zmq",
        num_agents=AGENT_NUM,
        game_id=game_id_init,
        single_test=single_test,
        data_shapes=config.data_shapes,
        lstm_time_steps=config.LSTM_TIME_STEPS,
        gamma=config.GAMMA,
        lamda=config.LAMDA,
    )

    monitor_ip, monitor_port = monitor_server_addr.split(":")
    monitor_logger = logging.getLogger("monitor")
    monitor_logger.setLevel(logging.INFO)
    monitor_handler = InfluxdbMonitorHandler(monitor_ip, monitor_port)
    monitor_handler.setLevel(logging.INFO)
    monitor_logger.addHandler(monitor_handler)

    camp_iter = camp_iterator()

    actor = Actor(
        id=actor_id,
        agents=agents,
        monitor_logger=monitor_logger,
        camp_iter=camp_iter,
        max_episode=max_episode,
        is_train=config.IS_TRAIN,
        enemy_type=config.ENEMY_TYPE,
    )
    actor.set_sample_manager(sample_manager)
    actor.set_env(env)
    try:
        actor.run(load_models=[], eval_freq=config.EVAL_FREQ)
    finally:
        game_launcher.stop_game(runtime_id)


def main(_):
    FLAGS = flags.FLAGS
    mem_pool_addr_list = FLAGS.mem_pool_addr.strip().split(";")
    monitor_ip = mem_pool_addr_list[0].split(":")[0]
    monitor_server_addr = f"{monitor_ip}:8086"
    from common.config import Config

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
        os.getenv("ACTOR_RUNTIME_ID_PREFIX", "actor-1v1"),
        "must not contain '_'",
    )
    flags.DEFINE_integer("port_begin", int(os.getenv("ACTOR_PORT_BEGIN", "35300")), "")
    flags.DEFINE_integer("max_frame_num", int(os.getenv("MAX_FRAME_NUM", "20000")), "")
    flags.DEFINE_boolean("debug_log", False, "use debug log level")
    absl_app.run(main)
