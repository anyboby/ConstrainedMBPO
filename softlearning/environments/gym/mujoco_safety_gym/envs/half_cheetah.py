import numpy as np
from gym import utils
from mujoco_safety_gym.envs import mujoco_env
import mujoco_py as mjp
from gym import error, spaces

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        
    def step(self, action):
        xposbefore = self.sim.data.qpos[1]
        
        t = self.data.time
        wall_act = .02*np.sin(t)**2 - .004
        mjp.functions.mj_rnePostConstraint(self.sim.model, self.sim.data) #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        action = np.concatenate((np.squeeze(action), [wall_act]))

        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[1]

        wallpos = self.data.get_geom_xpos("obj_geom")[0]
        wallvel = self.data.get_body_xvelp("obj1")[0]

        xdist = wallpos-xposafter
        obj_cost = int(np.abs(xdist)<2)

        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, cost=obj_cost)

    def _get_obs(self):
        wallvel = self.data.get_body_xvelp("obj1")[0]
        wall_f = .02*np.sin(self.data.time)**2 - .004
        xdist = self.data.get_geom_xpos("obj_geom")[0]-self.sim.data.qpos[1]
        
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat[1:],
            [wallvel],
            [wall_f],
            [xdist],
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low, high = low[:-1], high[:-1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
