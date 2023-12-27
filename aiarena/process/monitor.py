import os
import requests
from yaml import full_load
from process_base import ProcessBase
from hok.common.log import logger as LOG


class InfluxdbExporterProcess(ProcessBase):
    def __init__(
        self,
        port=8086,
        log_file="/aiarena/logs/influxdb-exporter.log",
    ) -> None:
        super().__init__(log_file)
        self.port = port

    def get_cmd_cwd(self):
        full_cmd = [
            "influxdb_exporter",
            "--web.listen-address",
            f":{self.port}",
            "--udp.bind-address",
            f":{self.port}",
        ]

        return full_cmd, None


class InfluxdbProcess(ProcessBase):
    def __init__(
        self,
        port=8086,
        retry_num=5,
        log_file="/aiarena/logs/influxdb.log",
    ) -> None:
        super().__init__(log_file)
        self.port = port
        self.retry_num = retry_num

    def get_cmd_cwd(self):
        full_cmd = ["/usr/bin/influxd"]
        return full_cmd, None

    def start(self):
        super().start()

        # wait server and create database
        self.wait_server_started("127.0.0.1", self.port)

        for i in range(self.retry_num):
            try:
                resp = requests.post(
                    f"http://127.0.0.1:{self.port}/query",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={"q": "CREATE DATABASE monitordb"},
                )
                resp.raise_for_status()
                break
            except Exception:
                LOG.exception(f"create influxdb database failed: {i}/{self.retry_num}")
                if i == self.retry_num - 1:
                    raise

        LOG.debug("influxd started")


class GrafanaServerProcess(ProcessBase):
    def __init__(self, log_file="/aiarena/logs/grafana.log") -> None:
        super().__init__(log_file)

    def get_cmd_cwd(self):
        full_cmd = [
            "/usr/sbin/grafana-server",
            "--config",
            "/etc/grafana/grafana.ini",
            "cfg:default.paths.provisioning=/etc/grafana/provisioning",
        ]
        cwd = "/usr/share/grafana"
        return full_cmd, cwd


if __name__ == "__main__":
    # influxdb_process = InfluxdbProcess()
    # influxdb_process.start()
    # influxdb_process.wait()
    # influxdb_exporter_process = InfluxdbExporterProcess()
    # influxdb_exporter_process.start()
    # influxdb_exporter_process.wait()
    grafana_process = GrafanaServerProcess()
    grafana_process.start()
    grafana_process.wait()
