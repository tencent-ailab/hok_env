from hok.hok3v3.agent import Agent as BaseAgent
from numpy.random import rand
import numpy as np

pred_ret_shape = [(1, 162), (1, 162), (1, 162)]
lstm_cell_shape = [(1, 16), (1, 16)]


class RandomAgent(BaseAgent):
    """
    random agent
    """

    def __init__(
        self, model_cls, model_pool_addr, keep_latest=False, local_mode=False, **kwargs
    ):
        super().__init__(
            model_cls, model_pool_addr, keep_latest, local_mode, rule_only=True
        )

    def _predict_process(self, hero_data_list, frame_state, runtime_ids):
        pred_ret = []
        for shape in pred_ret_shape:
            pred_ret.append(rand(*shape).astype("float32"))
        lstm_info = []
        for shape in lstm_cell_shape:
            lstm_info.append(np.zeros(shape).astype("float32"))

        return pred_ret, lstm_info
