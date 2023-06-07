# use default agent:
# from hok.hok3v3.agent import Agent as Agent

# custom agent
from hok.hok3v3.agent import Agent as BaseAgent
import json


class Agent(BaseAgent):
    def __init__(
        self, model_cls, model_pool_addr, keep_latest=False, local_mode=False, **kwargs
    ):
        super().__init__(model_cls, model_pool_addr, keep_latest, local_mode, **kwargs)

    def _predict_process(self, hero_data_list, frame_state, runtime_ids):
        if frame_state.frameNo // 3 == 100:
            print(json.dumps(frame_state.to_json()))
        pred_ret, lstm_info = super()._predict_process(
            hero_data_list, frame_state, runtime_ids
        )
        return pred_ret, lstm_info
