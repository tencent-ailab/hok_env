from process_base import ProcessBase


# TODO impl check_and_send
class CheckAndSendProcess(ProcessBase):
    def __init__(self, log_file="/aiarena/logs/send.log") -> None:
        super().__init__(log_file)

    def get_cmd_cwd(self):
        return super().get_cmd_cwd()
