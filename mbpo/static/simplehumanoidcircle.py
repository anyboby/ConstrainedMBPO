import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)
        z = next_obs[..., 2]
        notdone = np.isfinite(next_obs).all(axis=-1) \
            * (z >= 0.8) \
            * (z <= 2.0)

        done = ~notdone
        done = done[...,None]
        return done

    @staticmethod
    def cost_f(obs, act, next_obs, env):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        cost = np.abs(next_obs[...,-3]) >= 3
        cost = cost.astype(np.float32)
        cost = cost[...,None]
        return cost
