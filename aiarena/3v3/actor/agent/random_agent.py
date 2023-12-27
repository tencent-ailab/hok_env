from agent.agent import Agent as BaseAgent
from numpy.random import rand
import numpy as np

pred_ret_shape = [(1, 162)] * 3
lstm_cell_shape = [(1, 16), (1, 16)]


class Agent(BaseAgent):
    """
    random agent
    """

    def __init__(self, *args, **kwargs):
        kwargs["rule_only"] = True
        super().__init__(*args, **kwargs)

    def _predict_process(self, features, frame_state, runtime_ids):
        pred_ret = []
        for shape in pred_ret_shape:
            pred_ret.append(rand(*shape).astype("float32"))
        lstm_info = []
        for shape in lstm_cell_shape:
            lstm_info.append(np.zeros(shape).astype("float32"))

        return pred_ret, lstm_info
