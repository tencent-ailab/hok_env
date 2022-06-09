import sys

import numpy as np
import tensorflow as tf
from rl_framework.predictor.model_convertor import CkptToSavedModelConvertor


def get_pbtxt(input_tensor_list, output_tensor_list):
    # get config pbtxt
    def parse_type(dtype):
        if dtype == tf.float32:
            return "TYPE_FP32"
        elif dtype == tf.int32:
            return "TYPE_INT32"
        elif dtype == tf.int64:
            return "TYPE_INT64"
        elif dtype == tf.bool:
            return "TYPE_BOOL"
        else:
            raise NotImplementedError

    input_list = []
    output_list = []
    for tensor in input_tensor_list:
        data = {}
        data["name"] = '"%s"' % tensor.name
        data["data_type"] = parse_type(tensor.dtype)
        data["dims"] = tensor.shape.as_list()
        if len(data["dims"]) == 1:
            data["dims"] = [1]
            data["reshape"] = "{ shape: [ ] }"
        input_list.append(data)
    for tensor in output_tensor_list:
        data = {}
        data["name"] = '"%s"' % tensor.name
        data["data_type"] = parse_type(tensor.dtype)
        data["dims"] = tensor.shape.as_list()
        if len(data["dims"]) == 1:
            data["dims"] = [1]
            data["reshape"] = "{ shape: [ ] }"
        output_list.append(data)
    with open("config.pbtxt", "w") as f:
        f.write("max_batch_size: 256\n")
        f.write('platform: "tensorflow_savedmodel"\n')
        f.write("input:\n")
        f.write(("%s\n" % input_list).replace("'", ""))
        f.write("output:\n")
        f.write(("%s\n" % output_list).replace("'", ""))
        f.write("dynamic_batching {}\n")
        f.write("instance_group [ { count: 2 }]\n")


def main():
    file_from = sys.argv[1]
    file_to = sys.argv[2]
    # sys.path.append("/data1/reinforcement_platform/rl_learner_platform/code")
    # from mahjong_ppo.model import Model
    # sys.path.append("./model_code/")
    from model_code.model import Model

    model = Model()
    config_pbtxt_path = "./model_code/config.pbtxt"
    model_convertor = CkptToSavedModelConvertor(model, config_pbtxt_path)
    model_convertor.run(file_from, file_to)

    get_pbtxt(model_convertor._input_tensor_list, model_convertor._output_tensor_list)

    sys.exit(0)
    feature = np.random.rand(3, 734)
    legal_action = np.ones((3, 238))
    # print (feature, legal_action)
    result = model_convertor.inference(feature, legal_action)
    print(result)
    result2 = model_convertor.inference2(feature, legal_action)
    print(result2)


if __name__ == "__main__":
    main()
