from hok.hok3v3.agent import Agent as BaseAgent


class Agent(BaseAgent):
    def __init__(
        self, model_cls, model_pool_addr, keep_latest=False, local_mode=False, **kwargs
    ):
        super().__init__(
            model_cls, model_pool_addr, keep_latest, local_mode, rule_only=True
        )

    def reset(self, *args, **kwargs):
        return super().reset(agent_type="common_ai")
