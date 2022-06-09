# from SailLearner.algorithms.base.algorithm import Algorithm as AlgorithmBase
import threading
import sys

import numpy as np
import tensorflow as tf
from common_config import DimConfig
from common_config import ModelConfig as Config
from algorithm import Algorithm


class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            with Singleton._instance_lock:
                if self._cls not in self._instance:
                    self._instance[self._cls] = self._cls()
        return self._instance[self._cls]


# Singleton Pattern
@Singleton
class Model(Algorithm):
    def __init__(self):
        super().__init__()
        self.batch_size = 1
        # for actor:
        self.lstm_time_steps = 1
        self.batch_size = 1

        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        # self.hero_size = Config.HERO_SIZE
        # self.config_id_fea = Config.CONFIG_ID_FEA
        self.cut_points = [value[0] for value in Config.data_shapes]

        # Only True when evaluation
        self.deterministic_sample = False
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST  
        # print("legal_action size", self.legal_action_size)

        # net dims
        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(
            Config.LEGAL_ACTION_SIZE_LIST
        ) 
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.graph = None
