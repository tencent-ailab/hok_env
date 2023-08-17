import os

import sys


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.tools import freeze_graph
from model.tensorflow.model import Model
from config.model_config import ModelConfig


cpu_num = 1
sess_config = tf.ConfigProto(
    device_count={"CPU": cpu_num},
    inter_op_parallelism_threads=cpu_num,
    intra_op_parallelism_threads=cpu_num,
    log_device_placement=False,
)


def save_as_pb(
    graph, checkpoint_path, output_tensors, directory="checkpoints", filename="frozen"
):
    os.makedirs(directory, exist_ok=True)

    pbtxt_filename = filename + ".pbtxt"
    pbtxt_filepath = os.path.join(directory, pbtxt_filename)
    pb_filepath = os.path.join(directory, filename + ".pb")

    with tf.Session(graph=graph, config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.write_graph(
            graph_or_graph_def=sess.graph_def,
            logdir=directory,
            name=pbtxt_filename,
            as_text=True,
        )

    freeze_graph.freeze_graph(
        input_graph=pbtxt_filepath,
        input_saver="",
        input_binary=False,
        input_checkpoint=checkpoint_path,
        output_node_names=",".join([t.op.name for t in output_tensors]),
        restore_op_name="Unused",
        filename_tensor_name="Unused",
        output_graph=pb_filepath,
        clear_devices=True,
        initializer_nodes="",
    )

    return pb_filepath


model = Model(ModelConfig)
graph = model.build_infer_graph()

saver = tf.train.Saver(
    graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES),
    allow_empty=True,
)

checkpoint_path = "/aiarena/checkpoints/"
with tf.Session(graph=graph, config=sess_config) as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

output_tensors = model.get_output_tensors()
save_as_pb(graph, checkpoint_path + "model.ckpt", output_tensors, directory=checkpoint_path)
