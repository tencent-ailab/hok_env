import sys
from multiprocessing import Process

sys.path.append("/")
sys.path.append("/aiarena/code/actor/")  # TODO refactor


from aiarena.code.actor.entry import run
from aiarena.process.process_base import PyProcessBase


class ActorProcess(PyProcessBase):
    def __init__(
        self,
        actor_id=0,
        config_path=None,
        model_pool_addr="localhost:10016",
        single_test=False,
        port_begin=35300,
        gc_server_addr="127.0.0.1:23432",
        gamecore_req_timeout=30000,
        max_frame_num=20000,
        runtime_id_prefix="actor",
        aiserver_ip="127.0.0.1",
        mem_pool_addr_list=None,
        max_episode=-1,
        monitor_server_addr="127.0.0.1:8086",
        config=None,
        model_config=None,
    ) -> None:
        super().__init__()

        self.actor_id = actor_id
        self.config_path = config_path
        self.model_pool_addr = model_pool_addr
        self.single_test = single_test
        self.port_begin = port_begin
        self.gc_server_addr = gc_server_addr
        self.gamecore_req_timeout = gamecore_req_timeout
        self.max_frame_num = max_frame_num
        self.runtime_id_prefix = runtime_id_prefix
        self.aiserver_ip = aiserver_ip
        self.mem_pool_addr_list = mem_pool_addr_list or ["localhost:35200"]
        self.max_episode = max_episode
        self.monitor_server_addr = monitor_server_addr
        self.config = config
        self.model_config = model_config

    def start(self):
        self.proc = Process(
            target=run,
            args=(
                self.actor_id,
                self.config_path,
                self.model_pool_addr,
                self.single_test,
                self.port_begin,
                self.gc_server_addr,
                self.gamecore_req_timeout,
                self.max_frame_num,
                self.runtime_id_prefix,
                self.aiserver_ip,
                self.mem_pool_addr_list,
                self.max_episode,
                self.monitor_server_addr,
                self.config,
                self.model_config,
            ),
        )
        self.proc.start()


if __name__ == "__main__":
    actor = ActorProcess(single_test=True, max_frame_num=100, max_episode=2)
    actor.start()
    actor.wait()
