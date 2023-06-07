import os
from numpy.random import rand
import numpy as np

from hok.hok3v3.agent import Agent as BaseAgent
from hok.hok3v3.gamecore_client import GameCoreClient as Environment

pred_ret_shape = [(1, 162), (1, 162), (1, 162)]
lstm_cell_shape = [(1, 16), (1, 16)]


class RandomAgent(BaseAgent):
    """
    random agent
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, model_pool_addr=None, model_cls=None, rule_only=True, **kwargs
        )

    def _predict_process(self, *args, **kwargs):
        pred_ret = []
        for shape in pred_ret_shape:
            pred_ret.append(rand(*shape).astype("float32"))
        lstm_info = []
        for shape in lstm_cell_shape:
            lstm_info.append(np.zeros(shape).astype("float32"))

        return pred_ret, lstm_info


class CommonAIAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, model_pool_addr=None, model_cls=None, rule_only=True, **kwargs
        )
        self.agent_type = "common_ai"

    def reset(self, *args, **kwargs):
        return super().reset(agent_type="common_ai")


def run_test():
    GC_SERVER_ADDR = os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1:23432")
    # please replace the *AI_SERVER_ADDR* with your ip address.
    AI_SERVER_ADDR = os.getenv("AI_SERVER_ADDR", "127.0.0.1")

    print(GC_SERVER_ADDR, AI_SERVER_ADDR)

    agents = [RandomAgent(), CommonAIAgent()]
    print("init agent")

    env = Environment(host=AI_SERVER_ADDR, seed=0, gc_server=GC_SERVER_ADDR)
    _, is_gameover = env.reset(agents, eval_mode=True)

    for _ in range(100):
        if is_gameover:
            break
        for i, agent in enumerate(agents):
            if agent.is_common_ai():
                continue
            skip_prediction, features, req_pb = env.step_feature(i)
            if skip_prediction == 0:
                continue
            prob, lstm_info = agent.predict_process(features, req_pb)
            sample = env.step_action(prob, features, req_pb, i, lstm_info)
            is_gameover = req_pb.gameover

    env.close_game(agents)
    print("close game")

if __name__ == "__main__":
    run_test()
