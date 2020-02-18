import safety_gym
import gym
import numpy as np
class PerturbedEnv:
    def __init__(self, env, std_inc=0.02):
        self. std_inc = std_inc
        self.env = env
        self.rollouts = 1

    def step(self, act):
        next_obs, rewards, terminals, info, sim_state = self.env.step(act)
        next_obs = next_obs + np.random.normal(size=next_obs.shape)*((1+self.std_inc)**self.rollouts-1)
        self.rollouts += 1
        return next_obs, rewards, terminals, info, sim_state
    def reset(self, sim_state):
        obs, sim_state = self.env.reset(state_config=sim_state)
        self.rollouts = 1
        return obs, sim_state
    
