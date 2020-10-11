import numpy as np
from gym import utils
from mujoco_safety_gym.envs import mujoco_env
import mujoco_py as mjp

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[3]
        t = self.data.time
        pos = (t + np.sin(t)) + 3
        self.data.set_mocap_pos('mocap1', [pos, 0, 0.5])

        mjp.functions.mj_rnePostConstraint(self.sim.model, self.sim.data) #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[3]

        mocapx = self.sim.data.qpos[0]
        xdist = mocapx-xposafter
        obj_cost = int(np.abs(xdist)<1.5)

        # body_quat = self.data.get_body_xquat('torso')
        # z_rot = 1-2*(body_quat[1]**2+body_quat[2]**2)
        # trip_cost = int(z_rot<0.8)*0.1
        # cost = np.clip(trip_cost + obj_cost, 0, 1)

        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, cost=obj_cost)

    def _get_obs(self):
        x = self.sim.data.qpos[3]
        mocapx = self.sim.data.qpos[0]
        mocvel = 1 + np.cos(self.data.time)
        mocacc = -np.sin(self.data.time)

        return np.concatenate([
            self.sim.data.qpos.flat[4:],
            self.sim.data.qvel.flat[3:],
            [mocvel],
            [mocacc],
            [mocapx-x],
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
