import datetime
import os
import sys

from loguru import logger

INFO = "INFO"
CRITICAL = "CRITICAL"
ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"
DEBUG = "DEBUG"

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
    # add new more
}


def setup_logger(filename=None, level=None):
    logger.remove()
    if filename:
        logger.add(sys.stdout, level=(level or "ERROR"))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.add(filename, rotation="50 MB", level=(level or "INFO"))
    else:
        logger.add(sys.stdout, level=(level or "INFO"))


# log_time
def log_time(text):
    def decorator(func):
        def wrapper(*args, **kws):
            start = datetime.datetime.now()
            result = func(*args, **kws)
            end = datetime.datetime.now()
            time = (end - start).seconds * 1000.0 + (end - start).microseconds / 1000.0
            if g_log_time.get(text) is None:
                g_log_time[text] = []
            g_log_time[text].append(time)
            return result

        return wrapper

    return decorator


# log_time_func
def log_time_func(text, end=False):
    if g_log_time.get(text) is None:
        g_log_time[text] = []
    now = datetime.datetime.now()
    if len(g_log_time[text]) > 0:
        start = g_log_time[text][-1]
        if not isinstance(start, float):
            t = (now - start).seconds * 1000.0 + (now - start).microseconds / 1000.0
            g_log_time[text][-1] = t
    if not end:
        g_log_time[text].append(now)
