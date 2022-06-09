# -*- coding:utf-8 -*-
import os
import sys

from tensorflow.python import pywrap_tensorflow

if len(sys.argv) != 2:
    print "please used %s %s" % ("ckpt_reader.py", "model_path")
    sys.exit(0)

model_dir = sys.argv[1]
checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print ("tensor_name: ", key)
    print (reader.get_tensor(key))
