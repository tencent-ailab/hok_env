import tempfile
import os
import yaml
from yaml import Loader
from process_base import ProcessBase

default_pkg_path = "/rl_framework/model_pool/pkg/model_pool_pkg/"
# TODO install pkg to python package


class ModelPoolProcess(ProcessBase):
    def __init__(
        self,
        role="gpu",
        master_ip="127.0.0.1",
        log_file="/aiarena/logs/model_pool.log",
        pkg_path=default_pkg_path,
    ):
        """
        pkg_path: model_pool pkg包的路径, 用于启动程序
        log_path: model_pool 进程启动后的日志输出路径
        """
        super().__init__(log_file)
        self.pkg_path = pkg_path
        self.ip = "127.0.0.1"  # TODO 确认是否需要配置成当前ip
        self.cluster_context = "default"
        self.role = role
        self.master_ip = master_ip

    def _get_config(self, role, master_ip):
        # load default config from file
        config_file = os.path.join(self.pkg_path, "config", f"trpc_go.yaml.{role}")
        with open(config_file) as f:
            config = yaml.load(f, Loader=Loader)

        for _, log_plugin in config.get("plugins", {}).get("log", {}).items():
            for _config in log_plugin:
                if _config.get("writer_config", {}).get("filename"):
                    _config["writer_config"]["filename"] = self.log_file

        # overwrite default config
        if role == "cpu":
            config["client"]["service"][0]["target"] = f"dns://{master_ip}:10013"
            config["modelpool"]["ip"] = self.ip
            config["modelpool"]["name"] = self.ip
            config["modelpool"]["cluster"] = self.cluster_context
        elif role == "gpu":
            config["modelpool"]["ip"] = self.ip
        else:
            raise Exception(f"Unknow role: {role}")
        return config

    def _generate_config_file(self, role, master_ip):
        config = self._get_config(role, master_ip)
        fd, file = tempfile.mkstemp()
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        return file

    def get_cmd_cwd(self):
        config_file = self._generate_config_file(self.role, self.master_ip)
        full_cmd = ["./modelpool", "-conf", config_file]
        cwd = os.path.join(self.pkg_path, "bin")
        return full_cmd, cwd


class ModelPoolProxyProcess(ProcessBase):
    def __init__(
        self,
        file_save_path="/mnt/ramdisk/model",
        log_file="/aiarena/logs/model_pool_proxy.log",
        pkg_path=default_pkg_path,
    ) -> None:
        """
        pkg_path: model_pool pkg包的路径, 用于启动程序
        file_save_path: model_pool 模型保存的路径
        log_path: model_pool 进程启动后的日志输出路径
        """
        super().__init__(log_file)

        self.pkg_path = pkg_path
        self.file_save_path = file_save_path

    def get_cmd_cwd(self):
        os.makedirs(self.file_save_path, exist_ok=True)
        full_cmd = ["./modelpool_proxy", "-fileSavePath", self.file_save_path]
        cwd = os.path.join(self.pkg_path, "bin")
        return full_cmd, cwd


def run():
    model_pool_process = ModelPoolProcess()
    model_pool_proxy_process = ModelPoolProxyProcess()
    model_pool_process.start()
    model_pool_proxy_process.start()

    model_pool_process.wait()
    model_pool_proxy_process.wait()


if __name__ == "__main__":
    run()
