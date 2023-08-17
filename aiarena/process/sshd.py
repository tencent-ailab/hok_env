from process_base import ProcessBase
import os
import subprocess


class SSHDProcess(ProcessBase):
    def __init__(
        self, port=36001, passwd="passwd", log_file="/aiarena/logs/sshd.log"
    ) -> None:
        super().__init__(log_file)
        self.port = port
        self.passwd = passwd

    def get_cmd_cwd(self):
        subprocess.run(
            "rm -rf /root/.ssh/authorized_keys && ssh-keygen -t rsa -N '' -f /root/.ssh/id_rsa <<<y",
            shell=True,
        )
        subprocess.run(f'echo "root:{self.passwd}" | chpasswd', shell=True)

        os.makedirs("/run/sshd/", exist_ok=True)
        full_cmd = ["/usr/sbin/sshd", "-D"]
        return full_cmd, None

    def start(self):
        super().start()
        self.wait_server_started("127.0.0.1", self.port)

    def ssh_copy_id(self, cwd, ip, passwd):
        subprocess.run(f"./ssh-copy-id.expect {ip} {passwd}", cwd=cwd)
