import hashlib
import os
import sys
from model_syn_base import ModelSynBase
from rl_framework.model_pool import ModelPoolAPIs
from rl_framework.common.logging import logger as LOG

class ModelSynModelPool(ModelSynBase):
    def __init__(self, address):
        self.model_pool_apis = ModelPoolAPIs(address.split(','))
        self.model_pool_apis.check_server_set_up()
        self.step=0

    def syn_model(self, model_path, model_key = None):
        model, local_md5 = self._read_model(model_path)
        if model is None:
            return False
        if model_key is None:
            key = model_path.split("/")[-1]
            # key="model_{}".format(self.step)
        else:
            key = model_key
        self.model_pool_apis.push_model(model=model, hyperparam=None, key=key,\
             md5sum=local_md5, save_file_name=key)
        self.step += 1
        LOG.info("success push model ", key)
        return True

    def _read_model(self, model_path):
        if not os.path.exists(model_path):
            model = None
            local_md5 = None
            return model, local_md5
        else:
            with open(model_path, "rb") as fin:
                model = fin.read()
            #local_md5 = hashlib.md5(model).hexdigest()
            local_md5 = None
        return model, local_md5
