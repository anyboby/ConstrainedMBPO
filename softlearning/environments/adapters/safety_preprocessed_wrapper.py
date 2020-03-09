import numpy as np 
from scipy import signal
import gym

import matplotlib.pyplot as plt


class SafetyPreprocessedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(SafetyPreprocessedEnv, self).__init__(env)
        self.action_space_ext = gym.spaces.Box(-1, 1, (env.robot.nu+3,), dtype=np.float32)       # extended action space for stacking
        self.b, self.a = signal.butter(3, 0.1)
        self.obs_replay_vy_real = []
        self.obs_replay_vy_filt = []
        self.obs_replay_acc_y_real = []
        self.obs_replay_acc_y_filt = []
        self.prev_obs = None
        self.remove_obs = [
            'accelerometer',
            'gyro'
        ]
        self.obs_flat_size = sum([np.prod(i.shape) for i in self.env.obs_space_dict.values()])
        self.obs_flat_size = self.obs_flat_size-sum([np.prod(self.env.obs_space_dict[i].shape) for i in self.remove_obs])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, ((self.obs_flat_size),), dtype=np.float32)  #manually set size
        self.obs_indices = {}
        k_size = 0
        offset = 0
        for k in sorted(self.env.obs_space_dict.keys()):
            if k not in self.remove_obs:
                k_size = np.prod(self.env.obs_space_dict[k].shape)
                self.obs_indices[k] = slice(offset,offset + k_size)
                offset += k_size


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        #self.obs_replay_vy = [observation[58]]
        observation = self.preprocess_obs(observation)
        self.prev_obs = observation
        #stacked_obs = np.concatenate((observation, self.prev_obs))
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.prev_obs_unprocessed = observation
        observation = self.preprocess_obs(observation)
        #stacked_obs = np.concatenate((observation, self.prev_obs))
        self.prev_obs = observation
        return self.observation(observation), reward, done, info

    def observation(self, observation):

        return observation


    def preprocess_obs(self, obs):
        ###--------- Ordered obs ----------###
        # acc = obs['accelerometer']
        # goal_dist = obs['goal_dist']
        # goal_lidar = obs['goal_lidar']
        # gyro = obs['gyro']
        # hazards_lidar = obs['hazards_lidar']
        # magnetometer = obs['magnetometer']
        # vases_lidar = obs['vases_lidar']
        # velo = obs['velocimeter']
        ###--------- Ordered obs ----------###
        
        ### some additional options for debugging
        # qpos = self.env.data.qpos.copy()
        # qvel = self.env.data.qvel.copy()

        ###--------- vy velocity filtering ---------###
        # self.obs_replay_vy_real.append(velo[1])
        # v_y_filtered = signal.filtfilt(self.b, self.a, np.array(self.obs_replay_vy_real), method='gust')
        # velo[1] = v_y_filtered[-1]
        # self.obs_replay_vy_filt.append(v_y_filtered[-1])
        ###--------- vy velocity filtering ---------###
        
        ### --------- acc-y approx ----------### 
        # padding = np.zeros(shape=(2,))
        # vy_aug = np.concatenate((padding, v_y_filtered), axis=0)
        # self.obs_replay_acc_y_real.append(acc[1])
        # acc_y_filt = np.gradient(vy_aug, 1)[-1]
        # self.obs_replay_acc_y_filt.append(acc_y_filt)
        # acc[1] = acc_y_filt
        ### --------- acc-y approx ----------### 

        flat_obs = np.zeros(self.obs_flat_size)
        for key, index in self.obs_indices.items():
            if key not in self.remove_obs:
                flat_obs[index] = obs[key].flat
        obs = flat_obs
        
        return obs
