from collections import defaultdict

import numpy as np
from gym.spaces import Box, Dict, Discrete
import pdb

from .simple_replay_pool import SimpleReplayPool
from .simple_replay_pool import normalize_observation_fields

class MjcStateReplayPool(SimpleReplayPool):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super(MjcStateReplayPool, self).__init__(
            observation_space, action_space, *args, **kwargs)
        sim_field = {
                'sim_states': {
                    'shape': (1, ),
                    'dtype': 'object'
                }
        }
        self.add_fields(sim_field)