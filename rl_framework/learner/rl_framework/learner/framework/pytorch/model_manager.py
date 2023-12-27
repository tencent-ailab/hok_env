import hashlib
import os
import torch
import datetime
from multiprocessing import Process, Queue
from rl_framework.common.logging import logger as LOG


class ModelManager(object):
    def __init__(
        self,
        push_to_modelpool,
        remote_addrs=None,
        save_checkpoint_dir: str = "/aiarena/checkpoints/",
        backup_checkpoint_dir: str = "/rl_framework/send_model/model",
        push_model_queuing_dir: str = "/mnt/ramdisk/checkpoints/",
        load_optimizer_state=True,
    ):
        if remote_addrs is None:
            remote_addrs = ["127.0.0.1:10013:10014"]
        self.remote_addrs = remote_addrs
        self._push_to_modelpool = push_to_modelpool
        self.load_optimizer_state = load_optimizer_state

        self.save_checkpoint_dir = save_checkpoint_dir
        self.backup_checkpoint_dir = backup_checkpoint_dir
        self.push_model_queuing_dir = push_model_queuing_dir

        if self._push_to_modelpool:
            self.model_queue = Queue(maxsize=100)
            pid = Process(target=self._push_to_model_pool, args=())
            pid.daemon = True
            pid.start()

    def print_variables(self, net, optimizer, step):
        LOG.info(net)

    def _backup_checkpoint(
        self, save_model_dir, backup_checkpoint_dir, step, touch_done=True
    ):
        os.makedirs(backup_checkpoint_dir, exist_ok=True)
        tem = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace("-", "")
            .replace(":", "")
        )
        temp_ckpt = f"checkpoints_{tem}_{step}"
        os.system(
            f"cp -r {save_model_dir} {temp_ckpt}; tar cf {backup_checkpoint_dir}/{temp_ckpt}.tar {temp_ckpt}; rm -rf {temp_ckpt}"
        )
        if touch_done:
            os.system(
                f"touch {temp_ckpt}.tar.done; mv {temp_ckpt}.tar.done {backup_checkpoint_dir}/{temp_ckpt}.tar.done"
            )
        return os.path.abspath(os.path.join(backup_checkpoint_dir, f"{temp_ckpt}.tar"))

    def _push_to_model_pool(self):
        from rl_framework.model_pool import ModelPoolAPIs

        self.model_pool_apis = ModelPoolAPIs(self.remote_addrs)
        self.model_pool_apis.check_server_set_up()
        self.step = 0
        while True:
            try:
                model_path = self.model_queue.get()
                if not os.path.exists(model_path):
                    LOG.warning("{} not exists!!", model_path)
                    continue

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

                os.remove(model_path)
            except Exception:
                LOG.exception("push to model pool failed")

    def restore_model_and_optimizer(self, net, optimizer, model_path):
        LOG.info("Loading checkpoint from {} ...", model_path)
        state_dict = torch.load(model_path, map_location="cpu")

        if self.load_optimizer_state and "optimizer_state_dict" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            LOG.info("Load optimizer_state_dict state success")

        missing_keys, unexpected_keys = net.load_state_dict(
            state_dict["network_state_dict"], strict=False
        )

        LOG.info(
            "Load network_state_dict success, missing_keys: {}, unexpected_keys: {}",
            missing_keys,
            unexpected_keys,
        )
        return state_dict.get("step", 0)

    def save_checkpoint(self, net, optimizer, step: int):
        # for model pool
        if self._push_to_modelpool:
            self._save_checkpoint(
                net,
                optimizer,
                self.save_checkpoint_dir,
                step,
                save_optimizer_state=False,
            )

            ckpt_tar_file = self._backup_checkpoint(
                self.save_checkpoint_dir,
                self.push_model_queuing_dir,
                step,
                touch_done=False,
            )

            self.model_queue.put(ckpt_tar_file)
            LOG.info("Push checkpoint_{} {} to model_pool", step, ckpt_tar_file)

        # for backup
        LOG.info("Saving checkpoint_{}", step)
        self._save_checkpoint(net, optimizer, self.save_checkpoint_dir, step)

        ckpt_tar_file = self._backup_checkpoint(
            self.save_checkpoint_dir, self.backup_checkpoint_dir, step
        )
        LOG.info("Backup checkpoint_{} {}", step, ckpt_tar_file)

    def _save_checkpoint(
        self,
        net,
        optimizer,
        checkpoint_dir: str,
        step: int,
        save_optimizer_state: bool = True,
    ):
        os.makedirs(checkpoint_dir, exist_ok=True)
        step = int(step)
        checkpoint_file = os.path.join(checkpoint_dir, "model.pth")
        if save_optimizer_state:
            torch.save(
                {
                    "network_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                },
                checkpoint_file,
            )
        else:
            torch.save(
                {
                    "network_state_dict": net.state_dict(),
                    "step": step,
                },
                checkpoint_file,
            )
