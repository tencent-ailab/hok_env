# -*- coding: utf-8 -*-
from rl_framework.predictor import InferInput, InferOutput


def cvt_tensor_to_infer_input(input_tensors):
    """
    Convert tensor list to infer input list.

    Parameters
    ----------
    input_tensors : list of tf.Tensor
        A list of input tensors.

    Returns
    ----------
    list of InferInput
        A list of input tensors.
    """
    infer_input_list = []
    for inp in input_tensors:
        name = inp.name
        shape = [-1 if x is None else x for x in inp.shape.as_list()]
        dtype = inp.dtype.as_numpy_dtype
        infer_input_list.append(InferInput(name, shape, dtype))

    return infer_input_list


def cvt_tensor_to_infer_output(output_tensors):
    """
    Convert tensor list to infer output list.

    Parameters
    ----------
    output_tensors : list of tf.Tensor
        A list of output tensors.

    Returns
    ----------
    list of InferOutput
        A list of output tensors.
    """
    infer_output_list = []
    for out in output_tensors:
        name = out.name
        shape = [-1 if x is None else x for x in out.shape.as_list()]
        dtype = out.dtype.as_numpy_dtype
        infer_output_list.append(InferOutput(name, shape, dtype))

    return infer_output_list
