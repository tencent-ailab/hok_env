import sys
import json
import os
import tempfile
from multiprocessing import Process
import configparser


sys.path.append("/")

from aiarena.code.learner.train import run, config_path
from aiarena.process.process_base import PyProcessBase
from rl_framework.learner.framework.common.config_control import ConfigControl


class LearnerProcess(PyProcessBase):
    def __init__(
        self,
        mem_pool_port_list=None,
        single_test=False,
        default_config_path=config_path,
        display_every=1,
        save_model_steps=2,
        max_steps=5,
        batch_size=1,
        store_max_sample=20,
        use_xla=False,
        model_config=None,
    ) -> None:
        super().__init__()
        self.mem_pool_port_list = mem_pool_port_list or []
        self.default_config_path = default_config_path
        self.single_test = single_test

        self.display_every = display_every
        self.save_model_steps = save_model_steps
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.store_max_sample = store_max_sample
        self.use_xla = use_xla

        self.model_config = model_config

    def _get_config(self):
        config = configparser.ConfigParser()
        config.read(self.default_config_path)

        # overwrite config
        config.set("main", "ports", json.dumps(self.mem_pool_port_list))
        config.set("main", "backend", self.model_config.backend)
        if self.model_config.backend == "pytorch":  # TODO
            config.set("main", "distributed_backend", "none")

        config.set("main", "display_every", str(self.display_every))
        config.set("main", "save_model_steps", str(self.save_model_steps))
        config.set("main", "max_steps", str(self.max_steps))
        config.set("main", "batch_size", str(self.batch_size))
        config.set("dataset", "store_max_sample", str(self.store_max_sample))
        config.set("model", "use_xla", str(self.use_xla))
        config.set("model", "use_init_model", str(self.model_config.use_init_model))

        return config

    def _generate_config_file(self):
        config = self._get_config()
        fd, file = tempfile.mkstemp()
        with os.fdopen(fd, "w") as f:
            config.write(f)
        return file

    def start(self):
        config_path = self._generate_config_file()
        config_manager = ConfigControl(config_path)

        self.proc = Process(
            target=run, args=(self.model_config, config_manager, self.single_test)
        )
        self.proc.start()


if __name__ == "__main__":
    learner = LearnerProcess(single_test=True)
    learner.start()
    learner.wait()
    print(learner.exitcode())
