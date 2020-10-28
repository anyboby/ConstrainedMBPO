import sys
import numpy as np
import pdb

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        z = next_obs[...,0]
        done = (z < 1.0) + (z > 2.0)

        done = done[...,None]
        return done

    @staticmethod
    def cost_f(obs, act, next_obs, env):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        z = next_obs[..., 0]
        done = (z < 1.0) + (z > 2.0)
        done = done[...,None]
        done_cost = done*1.0
        
        y_dist = next_obs[..., -1:]
        obj_cost = np.any(abs(y_dist)>3.2, axis=-1)[...,None]*1.0

        cost = np.clip(done_cost+obj_cost, 0, 1)
        return cost
