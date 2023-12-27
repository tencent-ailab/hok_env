import subprocess
import time
import socket
import os

from hok.common.log import logger as LOG


class ServerProcess:
    def __init__(self) -> None:
        self.proc = None
        self.addr, self.port = "", 0

    # Get server package and extract to server_path
    def _create_server_path(self, server, server_path, server_driver):
        # get dst_file
        dst_file = None
        if server_driver == "local_tar":
            dst_file = server
        elif server_driver == "url":
            os.makedirs(server_path, exist_ok=True)
            dst_file = os.path.join(server_path, "server.tgz")
            cmd = ["wget", "-O", dst_file, server]
            LOG.info(cmd)
            subprocess.run(cmd, env=os.environ, check=True)

        # decompress dst_file
        os.makedirs(server_path, exist_ok=True)
        cmd = ["tar", "-C", server_path, "-xf", dst_file]
        LOG.info(cmd)
        subprocess.run(cmd, env=os.environ, check=True)
        return server_path

    # Extract server files for latter start up
    def _extract_server_files(self, server, server_path, server_port, server_driver):
        if server_driver in ["local_tar", "url"]:
            server_path = self._create_server_path(server, server_path, server_driver)
        elif server_driver == "server":
            self.addr, self.port = server, server_port
            return None
        elif server_driver == "common_ai":
            self.addr, self.port = "", 0
            return None
        else:
            server_path = server

        return server_path

    # Start server process
    def start(
        self,
        server,
        server_path,
        server_port,
        server_log_path,
        server_driver,
    ):

        server_path = self._extract_server_files(
            server, server_path, server_port, server_driver
        )
        if not server_path:
            return

        os.makedirs(os.path.dirname(server_log_path), exist_ok=True)
        full_cmd = [
            "python",
            "code/actor/server.py",
            "--server_addr",
            "tcp://0.0.0.0:{}".format(server_port),
        ]
        LOG.info(server_path)
        LOG.info(full_cmd)

        # redirect sdtout/stderr to server_log_path
        f = open(server_log_path, "w")
        self.proc = subprocess.Popen(
            full_cmd,
            env=os.environ,
            stderr=subprocess.STDOUT,
            stdout=f,
            preexec_fn=os.setsid,
            bufsize=10240,
            cwd=server_path,
        )
        self.addr, self.port = "127.0.0.1", server_port

    def get_server_addr(self):
        if self.addr and self.port:
            return (self.addr, self.port)
        return None

    def _test_connect(self, host, port):
        with socket.socket(socket.AF_INET) as s:
            try:
                s.connect((host, port))
            except ConnectionRefusedError:
                return False
        return True

    def wait_server_started(self, timeout):
        if (not self.addr) or (not self.port):
            return

        end_time = time.time() + timeout
        while time.time() < end_time:
            if self._test_connect(self.addr, self.port):
                break
            if self.proc and self.proc.poll() is not None:
                break
            time.sleep(1)

    # Stop server process
    def stop(self):
        if not self.proc:
            return

        self.proc.kill()
        if self.proc.stdout:
            self.proc.stdout.close()
        if self.proc.stderr:
            self.proc.stderr.close()
