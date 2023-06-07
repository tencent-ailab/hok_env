import logging.config
import os


class CommonLogger:
    g_log_instance = None

    @staticmethod
    def init_config():
        # create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # set formatter
        CommonLogger.g_log_instance = logger

    @staticmethod
    def set_config(actor_id, log_path="/aiarena/logs/actor"):
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(filename)s[line:%(lineno)d]\
                %(module)s[%(funcName)s] %(thread)d %(message)s"
        )

        os.makedirs(log_path, exist_ok=True)
        # set file handler: file rotates with time
        filename = log_path + "/info_%d.log" % int(actor_id)
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
