import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        z = next_obs[..., 0]
        body_quat = next_obs[...,1:5]
        z_rot = 1-2*(body_quat[...,1]**2+body_quat[...,2]**2)




        notdone = np.isfinite(next_obs).all(axis=-1) \
            * (z >= 0.2) \
            * (z <= 1.0) \
            * z_rot >= -0.7

        done = ~notdone
        done = done[...,None]
        return done

    @staticmethod
    def cost_f(obs, act, next_obs, env):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        z = next_obs[..., 0]
        body_quat = next_obs[...,1:5]
        z_rot = 1-2*(body_quat[...,1]**2+body_quat[...,2]**2)
        y_walldist = next_obs[..., -2:]
        
        obj_cost = (abs(y_walldist)<1.6).any()*1.0

        notdone = np.isfinite(next_obs).all(axis=-1) \
            * (z >= 0.2) \
            * (z <= 1.0) \
            * z_rot >= -0.7

        done = ~notdone
        done = done[...,None]

        done_cost = done*1.0
        cost = np.clip(done_cost+obj_cost, 0, 1)
        return cost
