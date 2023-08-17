import numpy as np
import rl_framework.common.logging as LOG

from agent import Agent as AgentBase

HERO_ID_INDEX_DICT = {
    112: 0,
    121: 1,
    123: 2,
    131: 3,
    132: 4,
    133: 5,
    140: 6,
    141: 7,
    146: 8,
    150: 9,
    154: 10,
    157: 11,
    163: 12,
    169: 13,
    175: 14,
    182: 15,
    193: 16,
    199: 17,
    502: 18,
    513: 19,
}


class Agent(AgentBase):
    def append_hero_identity(self, state_dict):
        # hero identity feature (ont-hot)
        runtime_id = state_dict["player_id"]
        hero_id = None
        for hero in state_dict["req_pb"].hero_list:
            if hero.runtime_id == runtime_id:
                hero_id = hero.config_id

        if hero_id is None:
            raise Exception("can not find config_id for runtime_id")

        hero_id_vec = np.zeros(
            [
                len(HERO_ID_INDEX_DICT),
            ],
            dtype=np.float,
        )
        if HERO_ID_INDEX_DICT.get(hero_id) is not None:
            hero_id_vec[HERO_ID_INDEX_DICT[hero_id]] = 1
        else:
            LOG.debug("Unknown hero_id for network: %s" % hero_id)
        state_dict["observation"] = np.concatenate(
            (state_dict["observation"], hero_id_vec), axis=0
        )
        return state_dict

    def feature_post_process(self, state_dict):
        state_dict = self.append_hero_identity(state_dict)
        return state_dict
