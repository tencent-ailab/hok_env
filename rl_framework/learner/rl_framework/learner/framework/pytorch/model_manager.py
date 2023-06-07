import datetime
import hashlib
import os
from multiprocessing import Process, Queue


class ModelManager(object):
    def __init__(
        self,
        push_to_modelpool,
        remote_addrs=None,
    ):
        if remote_addrs is None:
            remote_addrs = ["127.0.0.1:10013:10014"]
        self.remote_addrs = remote_addrs
        self._push_to_modelpool = push_to_modelpool
        if self._push_to_modelpool:
            self.model_queue = Queue(maxsize=100)
            pid = Process(target=self._push_to_model_pool, args=())
            pid.start()

    def send_model(self, save_model_dir, send_model_dir):
        os.makedirs(send_model_dir, exist_ok=True)
        tem = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace("-", "")
            .replace(":", "")
        )
        temp_ckpt = "checkpoints_" + tem
        os.system(
            f"c_dir=`pwd`; cp -r {save_model_dir} {temp_ckpt}; tar cf {send_model_dir}/{temp_ckpt}.tar {temp_ckpt}; rm -rf {temp_ckpt}; cd $c_dir"
        )
        os.system(
            f"c_dir=`pwd`; touch {temp_ckpt}.tar.done; mv {temp_ckpt}.tar.done {send_model_dir}/{temp_ckpt}.tar.done; cd $c_dir"
        )
        if self._push_to_modelpool:
            self.model_queue.put("{}/{}.tar".format(send_model_dir, temp_ckpt))
        ret = 0
        msg = f"c_dir=`pwd`; cp -r {save_model_dir} {temp_ckpt}; tar cf {send_model_dir}/{temp_ckpt}.tar {temp_ckpt}; rm -rf {temp_ckpt}; cd $c_dir"
        return ret, msg

    def _push_to_model_pool(self):
        from rl_framework.model_pool import ModelPoolAPIs

        self.model_pool_apis = ModelPoolAPIs(self.remote_addrs)
        self.model_pool_apis.check_server_set_up()
        self.step = 0
        while True:
            model_path = self.model_queue.get()
            if not os.path.exists(model_path):
                print("[model manager] {} not exists!!".format(model_path))
            else:
                with open(model_path, "rb") as fin:
                    model = fin.read()
                local_md5 = hashlib.md5(model).hexdigest()
                self.model_pool_apis.push_model(
                    model=model,
                    hyperparam=None,
                    key="model_{}".format(self.step),
                    md5sum=local_md5,
                    save_file_name=model_path.split("/")[-1],
                )
                self.step += 1
