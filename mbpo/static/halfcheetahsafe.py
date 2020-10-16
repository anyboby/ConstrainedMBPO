import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        done = np.array([False]).repeat(len(obs))
        done = done[...,None]
        return done


    @staticmethod
    def cost_f(obs, act, next_obs, env):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        xdist = next_obs[...,-1]*10
        obj_cost = np.array((np.abs(xdist)<2.0), dtype=np.float32)[..., None]
        return obj_cost
