import safety_gym
import gym
import numpy as np
class PerturbedEnv:
    def __init__(self, env, std_inc=0.02):
        self.std_inc = std_inc
        self.env = env
        self.rollouts = 1

    def step(self, act):
        next_obs, rewards, terminals, info = self.env.step(act)
        next_obs = next_obs + np.random.normal(size=next_obs.shape)*(self.std_inc*self.rollouts)
        self.rollouts += 1
        return next_obs, rewards, terminals, info

    def reset(self, sim_state=None):
        obs = self.env.reset(state_config=sim_state)
        self.rollouts = 1
        return obs
    
    def get_sim_state(self, *args, **kwargs):
        assert hasattr(self.env, 'get_sim_state')
        return self.env.get_sim_state(*args, **kwargs)
    
