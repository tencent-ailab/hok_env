import logging
import io
import time
from numpy import random

try:
    from modelPoolClient import ModelPoolClient
except Exception:  # pylint: disable=broad-except
    from rl_framework.model_pool.model_pool_api.modelPoolClient import ModelPoolClient


class ModelInfo:
    def __init__(
        self,
        key=None,
        file_name=None,
        save_file_path=None,
        time_stamp_nano=None,
        size=None,
        meta=None,
    ):
        self._key = key
        self._file_name = file_name
        self._save_file_path = save_file_path
        self._time_stamp_nano = time_stamp_nano
        self._size = size
        # self._meta = meta

    def __str__(self):
        return "key:%s, file_name:%s, save_file_path:%s,time_stamp_nano:%s,size:%s" % (
            self._key,
            self._file_name,
            self._save_file_path,
            self._time_stamp_nano,
            self._size,
        )

    def model_delay(self):
        return time.time() - (self._time_stamp_nano / 1000000000)


class ModelPoolAPIs(object):
    def __init__(self, model_pool_addrs):
        self._model_pool_addrs = list(model_pool_addrs)
        rand_int = random.randint(0, len(self._model_pool_addrs))
        addr = self._model_pool_addrs[rand_int]
        addr_split = addr.split(":")
        if len(addr_split) == 2:
            ip, http_port = addr_split[0], addr_split[1]
        elif len(addr_split) == 3:
            ip, http_port = addr_split[0], addr_split[2]
        else:
            logging.error("model_pool_addr error: %s" % addr)
            raise NotImplementedError
        self._model_pool_addr = "%s:%s" % (ip, http_port)
        self._client = ModelPoolClient(self._model_pool_addr)

    def check_server_set_up(self):
        pass

    # def push_model(self, key, meta):
    def push_model(
        self,
        key,
        model,
        hyperparam=None,
        createtime=None,
        freezetime=None,
        updatetime=None,
        learner_meta=None,
        md5sum=None,
        save_file_name=None,
    ):
        client = ModelPoolClient(self._model_pool_addr)
        client.uploadBytes(model, extraKey=key, filename=save_file_name)
        client.delete(deleteNoKeys=True)

    def push_model_from_path(self, key, path):
        client = ModelPoolClient(self._model_pool_addr)
        client.upload(path, extraKey=key)
        client.delete(deleteNoKeys=True)

    def pull_keys(self, model_num=100):
        client = ModelPoolClient(self._model_pool_addr)
        rsp = client.getFileInfo(newest=model_num)
        keys = []
        for file in rsp.files:
            keys.append(file.extraKey)
        return keys

    def pull_model(self, key):
        client = ModelPoolClient(self._model_pool_addr)
        buf = io.BytesIO()
        client.download(buf, extraKey=key)
        return buf.getvalue()

    def pull_model_info(self, key):
        client = ModelPoolClient(self._model_pool_addr)
        rsp = client.getFileInfo(extraKey=key)
        if len(rsp.files) == 0:
            return None
        file = rsp.files[0]
        return ModelInfo(
            key=file.extraKey,
            file_name=file.filename,
            save_file_path=file.absPath,
            time_stamp_nano=file.timestampNano,
            size=file.size,
        )

    def pull_model_path(self, key):
        client = ModelPoolClient(self._model_pool_addr)
        rsp = client.getFileInfo(extraKey=key)
        if rsp is None:
            return None
        if len(rsp.files) == 0:
            return None
        return rsp.files[0].absPath
