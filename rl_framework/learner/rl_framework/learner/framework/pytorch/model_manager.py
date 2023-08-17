import hashlib
import os
import torch
import datetime
from multiprocessing import Process, Queue
import rl_framework.common.logging as LOG


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
        self.load_optimizer_state = False

        if self._push_to_modelpool:
            self.model_queue = Queue(maxsize=100)
            pid = Process(target=self._push_to_model_pool, args=())
            pid.daemon = True
            pid.start()

    def print_variables(self, net, optimizer, step):
        LOG.info(net)

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
                LOG.info("[model manager] {} not exists!!".format(model_path))
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

    def restore_model_and_optimizer(self, net, optimizer, model_path):
        LOG.info(f"Loading checkpoint from {model_path} ...")
        state_dict = torch.load(model_path, map_location='cpu')
        if self.load_optimizer_state:
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        missing_keys, unexpected_keys = net.load_state_dict(state_dict["network_state_dict"], strict=False)
        LOG.info(f"load ckpt success, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")
        return state_dict.get("step", 0)

    def save_checkpoint(self, net, optimizer, checkpoint_dir: str, step: int):
        os.makedirs(checkpoint_dir, exist_ok=True)
        step = int(step)
        checkpoint_file = os.path.join(checkpoint_dir, "model.pth")
        torch.save(
            {
                "network_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            },
            checkpoint_file,
        )
