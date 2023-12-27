import datetime
import hashlib
import os
from multiprocessing import Process, Queue

import tensorflow as tf
from rl_framework.common.logging import logger as LOG


class ModelManager(object):
    def __init__(
        self,
        push_to_modelpool,
        remote_addrs=None,
    ):
        if remote_addrs is None:
            remote_addrs = ["127.0.0.1:10013:10014"]
        self.remote_addrs = remote_addrs
        self.saver = None
        self._push_to_modelpool = push_to_modelpool
        if self._push_to_modelpool:
            self.model_queue = Queue(maxsize=100)
            pid = Process(target=self._push_to_model_pool, args=())
            pid.daemon = True
            pid.start()

    def init_saver(self):
        self.saver = tf.train.Saver(self._savable_variables(), max_to_keep=0)

    def _savable_variables(self):
        params = {}
        for variable in tf.global_variables():
            name = variable.name
            if (
                name.startswith("input_datas")
                or "tower_0" in name
                # or "global_step" in name
            ):
                continue
            else:
                # if not name.startswith('v0/') and not 'global_step' in name:
                #    name = 'v0/%s' % name
                if name.endswith(":0"):
                    name = name[:-2]
                params[name] = variable
        return params

    def print_variables(self, sess):
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        with open("./log/variables_info.txt", "w") as f_out:
            for key, var in zip(variable_names, values):
                f_out.write("variables: " + str(key) + "\n")
                f_out.write("weights: " + str(var) + "\n")

    def restore_model(self, sess, model_path="./model/init"):
        model_checkpoint_path = os.path.join(model_path, "model.ckpt")
        self.saver.restore(sess, model_checkpoint_path)

    def save_model(self, sess, save_dir, send_dir):
        os.makedirs(send_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, "model.ckpt")
        self.saver.save(sess, checkpoint_path)
        tem = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace("-", "")
            .replace(":", "")
        )
        new_ckpt = "checkpoints_" + tem
        os.system(
            "c_dir=`pwd`; cp -r %s %s; tar cf %s/%s.tar %s; rm -rf %s; cd $c_dir"
            % (save_dir, new_ckpt, send_dir, new_ckpt, new_ckpt, new_ckpt)
        )
        os.system(
            "c_dir=`pwd`; touch %s.tar.done; mv %s.tar.done %s/%s.tar.done; cd $c_dir"
            % (new_ckpt, new_ckpt, send_dir, new_ckpt)
        )
        if self._push_to_modelpool:
            self.model_queue.put("{}/{}.tar".format(send_dir, new_ckpt))
        ret = 0
        msg = "c_dir=`pwd`; cp -r %s %s; tar cf %s/%s.tar %s; rm -rf %s; cd $c_dir" % (
            save_dir,
            new_ckpt,
            send_dir,
            new_ckpt,
            new_ckpt,
            new_ckpt,
        )
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
