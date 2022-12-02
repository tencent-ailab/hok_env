import datetime
import hashlib
import os
from multiprocessing import Process, Queue

import tensorflow as tf
from tensorflow.python.platform import gfile


class ModelManager(object):
    def __init__(self, push_to_modelpool, remote_addrs=None):
        if remote_addrs is None:
            remote_addrs = ["127.0.0.1:10013:10014"]
        self.remote_addrs = remote_addrs
        self.save_model_dir = "./checkpoints"
        self.saver = tf.train.Saver(self._savable_variables(), max_to_keep=0)
        self._push_to_modelpool = push_to_modelpool
        if push_to_modelpool:
            self.model_queue = Queue(maxsize=100)
            pid = Process(target=self._push_to_model_pool, args=())
            pid.start()

    def _savable_variables(self):
        params = []
        for variable in tf.global_variables():
            if not variable.name.startswith("input_datas"):
                params.append(variable)
        return params

    def print_variables(self, sess):
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        with open("./log/variables_info.txt", "w") as f_out:
            for key, var in zip(variable_names, values):
                f_out.write("variables: " + str(key) + "\n")
                f_out.write("weights: " + str(var) + "\n")

    def restore_model(self, sess, model_path="./model/init"):
        ckpt = tf.train.get_checkpoint_state(model_path)
        model_checkpoint_path = ckpt.model_checkpoint_path
        self.saver.restore(sess, model_checkpoint_path)

    def save_model(self, sess, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        checkpoint_path = os.path.join(self.save_model_dir, "model.ckpt")
        if not gfile.Exists(self.save_model_dir):
            gfile.MakeDirs(self.save_model_dir)
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
            % (self.save_model_dir, new_ckpt, save_path, new_ckpt, new_ckpt, new_ckpt)
        )
        os.system(
            "c_dir=`pwd`; touch %s.tar.done; mv %s.tar.done %s/%s.tar.done; cd $c_dir"
            % (new_ckpt, new_ckpt, save_path, new_ckpt)
        )
        if self._push_to_modelpool:
            self.model_queue.put("{}/{}.tar".format(save_path, new_ckpt))
        ret = 0
        msg = "c_dir=`pwd`; cp -r %s %s; tar cf %s/%s.tar %s; rm -rf %s; cd $c_dir" % (
            self.save_model_dir,
            new_ckpt,
            save_path,
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
