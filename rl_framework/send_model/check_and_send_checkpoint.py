import subprocess

import os
import sys
import time
import glob
from collections import OrderedDict
from absl import app
from absl import flags
import shutil

from rl_framework.common.logging import logger as LOG

flags.DEFINE_string("address", None, "remote server address")
flags.DEFINE_string("syn_type", "no_sync", "p2p or model_pool")
flags.DEFINE_boolean(
    "delete_checkpoint", os.getenv("DELETE_CKPT", "1") == "1", "delete the ckpt or not"
)
flags.DEFINE_boolean(
    "enable_backup", os.getenv("ENABLE_SEND_MODEL_BACKUP", "0") == "1", "enable backup"
)
flags.DEFINE_string("task_id", os.getenv("TASK_ID", "12345"), "task id")
flags.DEFINE_integer(
    "bkup_interval", int(os.getenv("BKUP_INTERVAL", "1800")), "model backup internal"
)
flags.DEFINE_string(
    "backup_base_dir",
    os.getenv("BACKUP_BASE_DIR", "/mnt/ceph/Model"),
    "model backup base dir",
)

flags.DEFINE_integer(
    "ckpt_file_storage_most_count",
    int(os.getenv("CKPT_FILE_STORAGE_MOST_COUNT", "20")),
    "model backup internal",
)

flags.DEFINE_string(
    "ckpt_dir",
    os.getenv(
        "CKPT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        # "/rl_framework/send_model/model"
    ),
    "model backup base dir",
)

flags.DEFINE_boolean(
    "backup_ckpt_only", os.getenv("BACKUP_CKPT_ONLY", "0") == "1", "Do not build_code, backup ckpt only"
)


class CheckAndSend:
    def __init__(
        self,
        model_syn,
        is_delete=False,
        use_bkup=False,
        task_id="",
        bkup_model_interval=60 * 30,
        backup_base_dir="/mnt/ceph/Model",
        ckpt_file_storage_most_count=20,
        ckpt_dir="/rl_framework/send_model/model",
        backup_ckpt_only=False,  # 仅备份ckpt文件, 不打包创建对战包
    ):
        self.model_syn = model_syn
        self.is_delete = is_delete
        self.task_id = task_id

        # bkup every 30 min.
        self.bkup_model_interval = bkup_model_interval
        self.use_bkup = use_bkup
        self.backup_base_dir = backup_base_dir
        self.ckpt_dir = ckpt_dir
        self.ckpt_file_storage_most_count = ckpt_file_storage_most_count
        self.backup_ckpt_only = backup_ckpt_only

    def run(self):
        files = glob.glob(self.ckpt_dir + "/*.tar.done")
        for file in files:
            os.remove(file)

        cp_file_order_dict = OrderedDict()
        send_file_done_order_dict = OrderedDict()
        the_last_model = None
        last_save_time = 0
        last_backup_name = ""

        while True:
            time.sleep(10)
            tmp_cp_file_list = sorted(
                glob.glob(self.ckpt_dir + "/*.tar.done"),
                key=os.path.getmtime,
            )
            tmp_cp_file_list = [x[:-5] for x in tmp_cp_file_list]
            if len(tmp_cp_file_list) > 0:
                model_file = tmp_cp_file_list[-1]
                model_name = model_file.split("/")[-1]

                if model_file not in cp_file_order_dict and (
                    the_last_model is None or model_name != the_last_model
                ):
                    self.model_syn.syn_model(model_file)

                the_last_model = model_name
                # save file info
                for tmp_cp_file in tmp_cp_file_list:
                    cp_file_order_dict[tmp_cp_file] = tmp_cp_file
                    send_file_done_order_dict[tmp_cp_file] = tmp_cp_file + ".done"

                # bkup model
                if self.use_bkup:
                    cur_time = time.time()
                    if (cur_time - last_save_time) > self.bkup_model_interval:
                        last_save_time = cur_time
                        need_del_ckpt_file_name = list(cp_file_order_dict.keys())[-1]
                        if last_backup_name != need_del_ckpt_file_name:
                            last_backup_name = need_del_ckpt_file_name
                            self._backup_file(
                                need_del_ckpt_file_name,
                                cp_file_order_dict,
                                self.ckpt_dir,
                            )

                if self.is_delete:
                    while len(cp_file_order_dict) > self.ckpt_file_storage_most_count:
                        need_del_ckpt_file_name = list(cp_file_order_dict.keys())[0]
                        self._delete_file(
                            need_del_ckpt_file_name,
                            cp_file_order_dict,
                            self.ckpt_dir,
                        )
                        self._delete_file(
                            need_del_ckpt_file_name,
                            send_file_done_order_dict,
                            self.ckpt_dir,
                        )
                        model_name = need_del_ckpt_file_name.split("/")[-1]
                        if model_name:
                            os.system("rm %s/%s*" % (self.ckpt_dir, model_name))

    def _backup_file(self, key_name, order_dict_obj, base_path):
        if self.backup_ckpt_only:
            return self._ckpt_backup_file(key_name, order_dict_obj, base_path)
        else:
            return self._build_code_backup_file(key_name, order_dict_obj, base_path)

    def _ckpt_backup_file(self, key_name, order_dict_obj, base_path):
        if key_name in order_dict_obj:
            dst_dir = os.path.join(self.backup_base_dir, self.task_id)
            os.makedirs(dst_dir, exist_ok=True)

            src_path = os.path.join(base_path, order_dict_obj[key_name])
            shutil.copy(src_path, dst_dir)
            LOG.info("backup model from %s to %s" % (src_path, dst_dir))

            subprocess.run(
                "cp /aiarena/logs/learner/loss.txt {}".format(dst_dir), shell=True
            )

    def _build_code_backup_file(self, key_name, order_dict_obj, base_path):
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

            LOG.info("backup model from %s to %s" % (src_path, dst_dir))
            sys.stdout.flush()

    def _delete_file(self, key_name, order_dict_obj, base_path):
        if key_name in order_dict_obj:
            os.system("cd %s; rm -r %s" % ((base_path), order_dict_obj[key_name]))
            LOG.info("%s has deleted" % (order_dict_obj[key_name]))
            sys.stdout.flush()
            del order_dict_obj[key_name]


def main(_):
    FLAGS = flags.FLAGS
    if FLAGS.syn_type == "model_pool":
        from model_syn_model_pool import ModelSynModelPool

        model_syn = ModelSynModelPool(FLAGS.address)
    elif FLAGS.syn_type == "p2p":
        from model_syn_p2p import ModelSynP2P

        model_syn = ModelSynP2P(FLAGS.address)
    elif FLAGS.syn_type == "no_sync":
        from model_no_syn import ModelSyn

        model_syn = ModelSyn()
    else:
        LOG.error("error! syn_type need model_pool or p2p")
        sys.exit(-1)

    check_and_send = CheckAndSend(
        model_syn,
        FLAGS.delete_checkpoint,
        FLAGS.enable_backup,
        task_id=FLAGS.task_id,
        bkup_model_interval=FLAGS.bkup_interval,
        backup_base_dir=FLAGS.backup_base_dir,
        ckpt_dir=FLAGS.ckpt_dir,
        ckpt_file_storage_most_count=FLAGS.ckpt_file_storage_most_count,
        backup_ckpt_only=FLAGS.backup_ckpt_only,
    )
    while True:
        try:
            check_and_send.run()
        except:
            LOG.exception("check and send failed")
        finally:
            time.sleep(1)


if __name__ == "__main__":
    app.run(main)
