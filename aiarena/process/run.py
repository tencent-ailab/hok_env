import sys
import time
import os
import socket

from absl import app

from rl_framework.common.logging import logger as LOG
from rl_framework.common.logging import setup_logger

# add aiarena
sys.path.append("/")

from aiarena.process.learner import LearnerProcess
from aiarena.process.model_pool import ModelPoolProcess, ModelPoolProxyProcess
from aiarena.process.monitor import (
    GrafanaServerProcess,
    InfluxdbExporterProcess,
    InfluxdbProcess,
)
from aiarena.process.send_model import CheckAndSendProcess
from aiarena.process.actor_process import ActorProcess
from aiarena.process.config_process import ConfigParser

setup_logger(filename="/aiarena/logs/run.log", level="INFO")

script_dir = os.path.dirname(os.path.abspath(__file__))

config_definition = {
    "input_learner_list": {
        "value": os.path.join(script_dir, "learner.iplist"),
        "help": "input learner list from platform",
        "env_alias": ["input_learner_list"],
    },
    "file_save_path": {
        "value": "/mnt/ramdisk/model",
        "env_alias": ["MODEL_POOL_FILE_SAVE_PATH"],
    },
    "not_use_influxdb_exporter": {
        "value": False,
        "env_alias": ["NOT_USE_INFLUXDB_EXPORTER"],
    },
    "net_card_name": {
        "value": "eth0",
        "env_alias": ["NET_CARD_NAME"],
    },
    "backend": {
        "value": "pytorch",
        "env_alias": ["AIARENA_BACKEND"],
    },
    "use_init_model": {
        "value": False,
        "env_alias": ["AIARENA_USE_INIT_MODEL"],
    },
    "use_ddp": {
        "value": True,
        "help": "enable ddp when run pytorch with multi learner",
    },  # TODO impl
    "actor_num": {
        "value": 1,
        "env_alias": ["CPU_NUM", "ACTOR_NUM"],
    },
    "max_episode": {
        "value": -1,
        "env_alias": ["MAX_EPISODE"],
    },
    "display_every": {
        "value": 1,
    },
    "save_model_steps": {
        "value": 2,
    },
    "max_steps": {
        "value": 5,
    },
    "batch_size": {
        "value": 1,
    },
    "store_max_sample": {
        "value": 20,
    },
    "use_xla": {
        "value": False,
    },
    "max_frame_num": {
        "value": 100,
    },
    "test_timeout": {
        "value": 300,
    },
    "game_mode": {
        "value": "1v1",
        "env_alias": ["CAMP_DEFAULT_MODE"],
    },
}


def run(config):
    def parse_iplist(learner_list):
        LOG.info("Parsing {}", learner_list)
        iplist = []
        with open(learner_list, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                vec = line.split()
                hostname = vec[0]
                LOG.debug("Parse {}", hostname)
                while True:
                    try:
                        ip = socket.gethostbyname(hostname)
                        break
                    except socket.error:
                        LOG.info(f"IP not found: {hostname}, retry...")
                        ip = hostname
                    time.sleep(1)
                vec[0] = ip
                vec[3] = int(vec[3])
                iplist.append(vec)
        return iplist

    LOG.info(config)

    input_learner_list = config["input_learner_list"]
    not_use_influxdb_exporter = config["not_use_influxdb_exporter"]
    net_card_name = config["net_card_name"]
    backend = config["backend"]
    use_init_model = config["use_init_model"]
    use_ddp = config["use_ddp"]
    actor_num = config["actor_num"]
    max_episode = config["max_episode"]
    display_every = config["display_every"]
    save_model_steps = config["save_model_steps"]
    max_steps = config["max_steps"]
    batch_size = config["batch_size"]
    store_max_sample = config["store_max_sample"]
    use_xla = config["use_xla"]
    max_frame_num = config["max_frame_num"]
    test_timeout = config["test_timeout"]

    learner_list = parse_iplist(input_learner_list)
    LOG.info("learner_list: {}", learner_list)

    master_ip = learner_list[0][0]
    node_list = [x[0] for x in learner_list]
    node_num = len(learner_list)
    learner_num = sum([x[3] for x in learner_list])
    proc_per_node = learner_num // node_num
    mem_pool_port_list = [35200 + i for i in range(proc_per_node)]
    mem_pool_addr_list = [f"{master_ip}:{port}" for port in mem_pool_port_list]
    monitor_port = 8086
    monitor_server_addr = f"{master_ip}:{monitor_port}"

    game_mode = config["game_mode"]
    if game_mode == "1v1":
        from aiarena.code.common.config import Config as CommonConfig

        actor_config = CommonConfig
        actor_model_config = None  # not used in 1v1
        learner_model_config = CommonConfig
        from hok.hok1v1.env1v1 import interface_default_config

    elif game_mode == "3v3":
        from aiarena.code.learner.config.Config import Config as learner_model_config
        from aiarena.code.actor.config.config import Config as actor_config
        from aiarena.code.actor.config.model_config import ModelConfig as actor_model_config
        from hok.hok3v3.hero_config import interface_default_config
    elif game_mode == "5v5dld":
        from aiarena.code.learner.config.Config import Config as learner_model_config
        from aiarena.code.actor.config.config import Config as actor_config
        from aiarena.code.actor.config.model_config import ModelConfig as actor_model_config
        from hok.hok5v5dld.hero_config import interface_default_config
    else:
        raise Exception(f"Unknown game_mode: {game_mode}")

    procs = []

    # TODO support cluster mode
    model_pool = ModelPoolProcess(
        role="gpu",  # TODO config role
        master_ip=master_ip,
    )
    model_pool_proxy = ModelPoolProxyProcess(
        file_save_path="/mnt/ramdisk/model",
    )

    procs.append(model_pool)
    procs.append(model_pool_proxy)

    if not_use_influxdb_exporter:
        influxdb = InfluxdbProcess(port=monitor_port)
        grafana = GrafanaServerProcess()
        procs.append(influxdb)
        procs.append(grafana)
    else:
        influxdb_exporter = InfluxdbExporterProcess(port=monitor_port)
        procs.append(influxdb_exporter)

    actor_config.backend = backend
    learner_model_config.backend = backend
    learner_model_config.use_init_model = use_init_model
    if backend != "pytorch" and not use_ddp:
        raise Exception("TODO impl")
    else:
        learner = LearnerProcess(
            mem_pool_port_list=mem_pool_port_list,
            display_every=display_every,
            save_model_steps=save_model_steps,
            max_steps=max_steps,
            batch_size=batch_size,
            store_max_sample=store_max_sample,
            use_xla=use_xla,
            model_config=learner_model_config,
        )
        procs.append(learner)

    actors = []
    for actor_id in range(actor_num):
        actor = ActorProcess(
            actor_id=actor_id,
            config_path=interface_default_config,
            aiserver_ip=master_ip,
            mem_pool_addr_list=mem_pool_addr_list,
            max_episode=max_episode,
            monitor_server_addr=monitor_server_addr,
            config=actor_config,
            model_config=actor_model_config,
            max_frame_num=max_frame_num,
        )
        procs.append(actor)
        actors.append(actor)

    # check_and_send = CheckAndSendProcess()  # TODO
    # procs.append(check_and_send)
    # TODO start gamecore server
    # TODO start monitor actor

    # start process
    try:
        for proc in procs:
            proc.start()

        LOG.info("All processes started, wait...")

        # wait learner
        actor_ended = False
        end_time = time.time() + test_timeout
        while not actor_ended or time.time() >= end_time:
            learner.wait(5)
            if learner.exitcode() is not None:
                break

            for actor in actors:
                if actor.exitcode() is not None:
                    LOG.warning("actor exit")
                    actor_ended = True
                    break
    finally:
        LOG.info("Stop all processes started, wait...")
        # 只有actor需要释放gamecore, 通过terminate触发
        for proc in actors:
            LOG.debug(f"terminate {proc}")
            proc.terminate()

        end_time = time.time() + 30
        for proc in actors:
            timeout = end_time - time.time()
            if timeout <= 0:
                LOG.warning("procs terminate timeout")
                break
            LOG.debug(f"wait {proc}")
            proc.wait(timeout)

        for proc in procs:
            LOG.debug(f"stop {proc}")
            proc.stop()

    if learner.exitcode() != 0:
        raise Exception(f"Test failed: learner exit non-zero: {learner.exitcode()}")

    LOG.info("Test success")


if __name__ == "__main__":
    parser = ConfigParser()
    parser.register_config_to_flags("aiarena", config_definition)

    def main(_):
        try:
            config = parser.parse("aiarena", config_definition)
            run(config)
        except Exception:
            LOG.exception("run failed")
            raise

    app.run(main)
