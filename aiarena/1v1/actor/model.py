from rl_framework.common.utils.common_func import Singleton


def get_model_class(backend):
    if backend == "pytorch":
        from common.algorithm_torch import Algorithm
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    elif backend == "tensorflow":
        from common.algorithm_tf import Algorithm

    # Singleton Pattern
    @Singleton
    class Model(Algorithm):
        def __init__(self):
            super().__init__()
            self.lstm_time_steps = 1
            self.batch_size = 1

    return Model
