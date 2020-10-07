import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

        x = next_obs[..., 0]
        obj_xy = next_obs[..., -24:]
        obj_dist_x = 1/obj_xy[...,0::2]
        obj_dist_y = 1/obj_xy[...,1::2]
        obj_dist = np.sqrt(obj_dist_x**2 + obj_dist_y**2)
        obj_cost = np.any(obj_dist<2.0, axis=-1)*1.0
        not_done = 	np.isfinite(next_obs).all(axis=-1) \
        			* (x >= 0.2) \
        			* (x <= 1.0) \
                    * obj_cost==0

        done = ~not_done
        done = done[...,None]
        return done
