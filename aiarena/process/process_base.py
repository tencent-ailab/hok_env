import time
import socket
import os
import subprocess
from rl_framework.common.logging import logger as LOG

class PyProcessBase:
    def __init__(self) -> None:
        self.proc = None

    def stop(self):
        if self.proc:
            self.proc.kill()

    def wait(self, timeout=None):
        if self.proc:
            self.proc.join(timeout=timeout)

    def terminate(self):
        if self.proc:
            self.proc.terminate()

    def exitcode(self):
        if self.proc:
            return self.proc.exitcode
        return None


class ProcessBase:
    def __init__(
        self,
        log_file="/aiarena/logs/process_base.log",
    ) -> None:
        self.log_file = log_file
        self.proc = None

    def get_cmd_cwd(self):
        return ["echo", "123"], "/"

    # Start process
    def start(self):
        # redirect sdtout/stderr to log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        f = open(self.log_file, "w")

        # _start_model_pool
        full_cmd, cwd = self.get_cmd_cwd()
        LOG.debug(f"start process: {cwd} {full_cmd}")
        self.proc = subprocess.Popen(
            full_cmd,
            env=os.environ,
            stderr=subprocess.STDOUT,
            stdout=f,
            preexec_fn=os.setsid,
            bufsize=10240,
            cwd=cwd,
        )

    # Stop process
    def stop(self):
        if not self.proc:
            return

        self.proc.kill()
        if self.proc.stdout:
            self.proc.stdout.close()
        if self.proc.stderr:
            self.proc.stderr.close()

    def wait(self, timeout=None):
        if not self.proc:
            return
        self.proc.wait(timeout)

    def _test_connect(self, host, port):
        with socket.socket(socket.AF_INET) as s:
            try:
                s.connect((host, port))
            except ConnectionRefusedError:
                return False
        return True

    def wait_server_started(self, host, port, timeout=-1):
        start_time = time.time()
        while timeout <= 0 or time.time() - start_time < timeout:
            if self._test_connect(host, port):
                break
            if self.proc and self.proc.poll() is not None:
                LOG.warning("proc terminated")
                break
            LOG.debug(f"server({host},{port}) not ok, wait")
            time.sleep(1)

    def terminate(self):
        if self.proc:
            self.proc.terminate()
