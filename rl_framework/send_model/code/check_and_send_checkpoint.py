from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import glob
from collections import OrderedDict
from absl import app
from absl import flags
import subprocess

FLAGS = flags.FLAGS
flags.DEFINE_string("address", None, "remote server address")
flags.DEFINE_string("syn_type", None, "p2p or model_pool")
flags.DEFINE_boolean("is_delete", False, "delete the model or not")
flags.DEFINE_string("predictor_type", "local", "local or remote")
flags.DEFINE_boolean("enable_backup", False, "enable backup")
flags.DEFINE_string("task_id", os.getenv("TASK_ID", ""), "task id")
flags.DEFINE_integer(
    "bkup_interval", int(os.getenv("BKUP_INTERVAL", "1800")), "model backup internal"
)
flags.DEFINE_string(
    "backup_base_dir",
    os.getenv("BACKUP_BASE_DIR", "/mnt/ceph/Model"),
    "model backup base dir",
)




class CheckAndSend(object):
    def __init__(
        self,
        mode_syn,
        is_delete=False,
        use_bkup=False,
        task_id="",
        bkup_model_interval=60 * 30,
        backup_base_dir="/mnt/ceph/Model",
    ):
        self.model_syn = mode_syn
        self.is_delete = is_delete
        self.task_id = task_id

        # bkup every 30 min.
        self.bkup_model_interval = bkup_model_interval
        self.use_bkup = use_bkup
        self.backup_base_dir = backup_base_dir

        if FLAGS.predictor_type == "remote":
            from model_code.model import Model
            from rl_framework.predictor.model_convertor import CkptToSavedModelConvertor

            model = Model()
            config_pbtxt_path = os.path.abspath("./model_code/config.pbtxt")
            self.convertor = CkptToSavedModelConvertor(model, config_pbtxt_path)

    def run(self):
        base_path = os.path.abspath("..")
        ckpt_file_storage_most_count = 200
        cp_file_order_dict = OrderedDict()
        send_file_done_order_dict = OrderedDict()
        the_last_model = None
        last_save_time = None

        while True:
            time.sleep(1)
            tmp_cp_file_list = sorted(
                glob.glob(base_path + "/model/*.tar.done"),
                key=lambda x: os.path.getmtime(x),
            )
            tmp_cp_file_list = [x[:-5] for x in tmp_cp_file_list]
            if (len(tmp_cp_file_list) > 0) and (
                tmp_cp_file_list[-1] not in cp_file_order_dict
            ):
                model_file = tmp_cp_file_list[-1]
                model_name = model_file.split("/")[-1]
                if the_last_model is not None and model_name == the_last_model:
                    continue

                if FLAGS.predictor_type == "local":
                    self.model_syn.syn_model(model_file)
                elif FLAGS.predictor_type == "remote":
                    convert_model_file = "%s.convert" % (model_file)
                    self.convertor.run(model_file, convert_model_file)
                    self.model_syn.syn_model(convert_model_file)
                else:
                    raise NotImplementedError

                the_last_model = model_name
                # save file info
                for tmp_cp_file in tmp_cp_file_list:
                    cp_file_order_dict[tmp_cp_file] = tmp_cp_file
                    send_file_done_order_dict[tmp_cp_file] = tmp_cp_file + ".done"

                # bkup model
                if self.use_bkup:
                    cur_time = time.time()
                    if (
                        last_save_time is None
                        or (cur_time - last_save_time) > self.bkup_model_interval
                    ):
                        last_save_time = cur_time
                        need_del_ckpt_file_name = list(cp_file_order_dict.keys())[-1]

                        self._backup_file(
                            need_del_ckpt_file_name,
                            cp_file_order_dict,
                            (base_path + "/model/"),
                        )

                if self.is_delete:
                    while len(cp_file_order_dict) > ckpt_file_storage_most_count:
                        need_del_ckpt_file_name = list(cp_file_order_dict.keys())[0]
                        self._delete_file(
                            need_del_ckpt_file_name,
                            cp_file_order_dict,
                            (base_path + "/model/"),
                        )
                        self._delete_file(
                            need_del_ckpt_file_name,
                            send_file_done_order_dict,
                            (base_path + "/model/"),
                        )
                        model_name = need_del_ckpt_file_name.split("/")[-1]
                        os.system("cd %s/model/; rm %s*" % (base_path, model_name))

    def _backup_file(self, key_name, order_dict_obj, base_path):

        if key_name in order_dict_obj:
            src_path = os.path.join(base_path, order_dict_obj[key_name])
            dst_dir = os.path.join(self.backup_base_dir, self.task_id)
            os.makedirs(dst_dir, exist_ok=True)
            os.makedirs("/aiarena/code/actor/model/init", exist_ok=True)

            # 解压模型/同步代码
            env = os.environ.copy()
            env["OUTPUT_DIR"] = dst_dir
            subprocess.run(
                "tar -C /aiarena/code/actor/model/init --strip-components=1 -xf {}".format(
                    src_path
                ),
                shell=True,
            )
            subprocess.run("sh /aiarena/scripts/build_code.sh", env=env, shell=True)
            subprocess.run(
                "cp /aiarena/logs/learner/loss.txt {}".format(dst_dir), shell=True
            )

            print("backup model from %s to %s" % (src_path, dst_dir))
            sys.stdout.flush()

    def _delete_file(self, key_name, order_dict_obj, base_path):
        if key_name in order_dict_obj:
            os.system("cd %s; rm -r %s" % ((base_path), order_dict_obj[key_name]))
            print("%s has deleted" % (order_dict_obj[key_name]))
            sys.stdout.flush()
            del order_dict_obj[key_name]


def main(_):
    print("address:{} syn_type:{}".format(FLAGS.address, FLAGS.syn_type))
    if FLAGS.address is None or FLAGS.syn_type is None:
        print("error! need params --address --syn_type")
        sys.exit(-1)
    if FLAGS.syn_type == "model_pool":
        from model_syn_model_pool import ModelSynModelPool

        model_syn = ModelSynModelPool(FLAGS.address)
    elif FLAGS.syn_type == "p2p":
        from model_syn_p2p import ModelSynP2P

        model_syn = ModelSynP2P(FLAGS.address)
    else:
        print("error! syn_type need model_pool or p2p")
        sys.exit(-1)
    check_and_send = CheckAndSend(
        model_syn,
        FLAGS.is_delete,
        FLAGS.enable_backup,
        task_id=FLAGS.task_id,
        bkup_model_interval=FLAGS.bkup_interval,
        backup_base_dir=FLAGS.backup_base_dir,
    )
    check_and_send.run()


if __name__ == "__main__":
    app.run(main)
