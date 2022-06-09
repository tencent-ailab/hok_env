from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import sys
import time
from collections import OrderedDict

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("address", None, "remote server address")
flags.DEFINE_string("syn_type", None, "p2p or model_pool")
flags.DEFINE_boolean("is_delete", False, "delete the model or not")
flags.DEFINE_string("predictor_type", "local", "local or remote")

from rl_framework.predictor.model_convertor import CkptToSavedModelConvertor


class CheckAndSend(object):
    def __init__(self, mode_syn, is_delete=False):
        self.model_syn = mode_syn
        self.is_delete = is_delete
        self.task_id = os.getenv("TASK_ID")

        # bkup every 30 min.
        self.bkup_model_interval = 60 * 10
        if os.getenv("BKUP_INTERVAL") is not None:
            self.bkup_model_interval = float(os.getenv("BKUP_INTERVAL").strip()) * 60
        self.use_bkup = True

        if FLAGS.predictor_type == "remote":
            from model_code.model import Model

            model = Model()
            config_pbtxt_path = os.path.abspath("./model_code/config.pbtxt")
            self.convertor = CkptToSavedModelConvertor(model, config_pbtxt_path)

    def run(self):
        base_path = os.path.abspath("..")
        ckpt_file_storage_most_count = 20
        cp_file_order_dict = OrderedDict()
        send_file_done_order_dict = OrderedDict()
        the_last_model = None
        last_save_time = None

        while True:
            tmp_cp_file_list = sorted(
                glob.glob(base_path + "/model/*.tar"), key=lambda x: os.path.getmtime(x)
            )
            time.sleep(1)
            if (len(tmp_cp_file_list) > 0) and (
                tmp_cp_file_list[-1] not in cp_file_order_dict
            ):
                time.sleep(0.2)
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
                        self._backup_file(
                            need_del_ckpt_file_name,
                            send_file_done_order_dict,
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
            os.system(
                "cd %s; cp %s /model_bkup/" % ((base_path), order_dict_obj[key_name])
            )
            os.system("cp /code/gpu_code/learner/log/loss.txt /model_bkup/")

            print("%s has backup to /model_bkup/" % (order_dict_obj[key_name]))
            sys.stdout.flush()
            # del order_dict_obj[key_name]

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
    check_and_send = CheckAndSend(model_syn, FLAGS.is_delete)
    check_and_send.run()


if __name__ == "__main__":
    app.run(main)
