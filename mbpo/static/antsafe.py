import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        x = next_obs[..., 0]
        obj_dists = next_obs[..., -24:]
        obj_cost = (obj_dists<2.0).any()*1.0
        not_done = 	np.isfinite(next_obs).all(axis=-1) \
        			* (x >= 0.2) \
        			* (x <= 1.0)\
                and obj_cost==0

        done = ~not_done
        done = done[...,None]
        return done
