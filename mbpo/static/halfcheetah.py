import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        done = np.zeros(shape=obs.shape[:-1], dtype=np.bool) #np.array([False]).repeat(len(obs))
        done = done[...,None]
        return done
