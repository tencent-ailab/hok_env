import os

# from model_pool_apis import ModelPoolAPIs
from rl_framework.model_pool import ModelPoolAPIs

from model_syn_base import ModelSynBase


class ModelSynModelPool(ModelSynBase):
    def __init__(self, address):
        self.model_pool_apis = ModelPoolAPIs(address.split(","))
        self.model_pool_apis.check_server_set_up()
        self.step = 0

    def syn_model(self, model_path):
        model, local_md5 = self._read_model(model_path)
        if model is None:
            return False
        key = model_path.split("/")[-1]
        self.model_pool_apis.push_model(
            model=model,
            hyperparam=None,
            key="model_{}".format(self.step),
            md5sum=local_md5,
            save_file_name=key,
        )
        # self.model_pool_apis.push_model(model=model, hyperparam=None, key=key,\
        #     md5sum=local_md5, save_file_name=key)
        self.step += 1
        print("success push model ", key)
        return True

    def _read_model(self, model_path):
        if not os.path.exists(model_path):
            model = None
            local_md5 = None
            return model, local_md5
        else:
            with open(model_path, "rb") as fin:
                model = fin.read()
            # local_md5 = hashlib.md5(model).hexdigest()
            local_md5 = None
        return model, local_md5
