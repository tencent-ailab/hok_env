from agent.agent import Agent as BaseAgent


class Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        kwargs["rule_only"] = True
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        return super().reset(agent_type="common_ai")
