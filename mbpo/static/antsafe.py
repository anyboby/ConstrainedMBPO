import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        x = next_obs[..., 0]

        body_quat = next_obs[...,1:5]
        z_rot = 1-2*(body_quat[...,1]**2+body_quat[...,2]**2)

        notdone = np.isfinite(next_obs).all(axis=-1) \
            * (x >= 0.2) \
            * (x <= 1.0) \
            * z_rot >= -0.7

        done = ~notdone
        done = done[...,None]
        return done

    @staticmethod
    def cost_f(obs, act, next_obs, env):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        x = next_obs[..., 0]
        obj_dist = next_obs[..., -2:]
        obj_cost = np.any(abs(obj_dist)<1.8, axis=-1)*1.0
        obj_cost = obj_cost[..., None]

        body_quat = next_obs[...,1:5]
        z_rot = 1-2*(body_quat[...,1]**2+body_quat[...,2]**2)

        notdone = np.isfinite(next_obs).all(axis=-1) \
            * (x >= 0.2) \
            * (x <= 1.0) \
            * z_rot >= -0.7

        done = ~notdone
        done = done[...,None]

        done_cost = done*1.0
        cost = np.clip(done_cost+obj_cost, 0, 1)
        return cost
