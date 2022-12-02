class AgentBase:
    def __init__(self) -> None:
        self.keep_latest = True

        self.player_id = None
        self.hero_camp = None

        self.hero_type = None
        self.is_latest_model = None
        self.agent_type = None

    def set_lstm_info(self, lstm_info):
        raise Exception("Not implemented")

    def get_lstm_info(self):
        raise Exception("Not implemented")

    def process(self, state_dict, battle=False):
        raise Exception("Not implemented")

    def reset(self, agent_type=None, model_path=None):
        raise Exception("Not implemented")

    def close(self):
        raise Exception("Not implemented")

    def set_game_info(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id
