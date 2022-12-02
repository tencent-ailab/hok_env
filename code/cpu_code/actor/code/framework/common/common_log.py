import logging.config

g_log_time = {
    "feature_process": [],
    "reward_process": [],
    "result_process": [],
    "predict_process": [],
    "aiprocess_process": [],
    "gamecore_process": [],
    "sample_manger_format_data": [],
    "send_data": [],
    "agent_process": [],
    "step": [],
    "save_sample": [],
    "one_frame": [],
    "one_episode": [],
    "reset": [],
    "step_af": [],
}


class CommonLogger:
    g_log_instance = None

    @staticmethod
    def init_config():
        # logging.config.fileConfig('./config/logging.conf')
        # CommonLogger.g_log_instance = logging.get_logger()

        # create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # set formatter
        CommonLogger.g_log_instance = logger

    @staticmethod
    def set_config(actor_id):
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(filename)s[line:%(lineno)d]\
                %(module)s[%(funcName)s] %(thread)d %(message)s"
        )

        # set file handler: file rotates with time
        filename = "/code/logs/cpu_log/info_%d.log" % int(actor_id)
        rf_handler = logging.handlers.TimedRotatingFileHandler(
            filename=filename, when="H", interval=3, backupCount=1
        )
        rf_handler.setLevel(logging.INFO)
        rf_handler.setFormatter(formatter)

        # set console handler: display on the console
        console = logging.StreamHandler()
        console.setLevel(logging.ERROR)
        console.setFormatter(formatter)

        CommonLogger.g_log_instance.addHandler(rf_handler)
        CommonLogger.g_log_instance.addHandler(console)

    @staticmethod
    def get_logger():
        if not CommonLogger.g_log_instance:
            CommonLogger.init_config()
        return CommonLogger.g_log_instance
