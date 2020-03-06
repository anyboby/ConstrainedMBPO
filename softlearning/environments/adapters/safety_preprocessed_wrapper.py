import numpy as np 
from scipy import signal
import gym

import matplotlib.pyplot as plt


class SafetyPreprocessedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(SafetyPreprocessedEnv, self).__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (env.obs_flat_size-6,), dtype=np.float32)  #removing gyro
        self.action_space_ext = gym.spaces.Box(-1, 1, (env.robot.nu+3,), dtype=np.float32)       # extended action space for stacking
        self.b, self.a = signal.butter(3, 0.1)
        self.obs_replay_vy_real = []
        self.obs_replay_vy_filt = []
        self.obs_replay_acc_y_real = []
        self.obs_replay_acc_y_filt = []

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        #self.obs_replay_vy = [observation[58]]
        observation = self.preprocess_obs(observation)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.preprocess_obs(observation)
        return self.observation(observation), reward, done, info

    def observation(self, observation):

        return observation


    def preprocess_obs(self, obs):
        acc = obs[0:3]
        goal_lidar = obs[3:19]
        gyro = obs[19:22]
        hazards_lidar = obs[22:38]
        magnetometer = obs[38:41]
        vases_lidar = obs[41:57]
        velo = obs[57:60]

        ### vy velocity filtering
        # self.obs_replay_vy_real.append(velo[1])
        # v_y_filtered = signal.filtfilt(self.b, self.a, np.array(self.obs_replay_vy_real), method='gust')
        # velo[1] = v_y_filtered[-1]
        # self.obs_replay_vy_filt.append(v_y_filtered[-1])

        ### acc-y 
        # padding = np.zeros(shape=(2,))
        # vy_aug = np.concatenate((padding, v_y_filtered), axis=0)
        # self.obs_replay_acc_y_real.append(acc[1])
        # acc_y_filt = np.gradient(vy_aug, 1)[-1]
        # self.obs_replay_acc_y_filt.append(acc_y_filt)
        # acc[1] = acc_y_filt

        # if len(self.obs_replay_vy)>400:
        #     plt.plot(np.arange(len(self.obs_replay_vy)), self.obs_replay_vy, color='blue')                         # filtered y-vel
        #     plt.plot(np.arange(len(self.obs_replay_vy)), self.obs_replay_vy_filt, color='red')                         # filtered y-vel

        #     plt.show()

        ### remove gyro
        # new_obs = np.concatenate((acc, goal_lidar, hazards_lidar, magnetometer, vases_lidar, velo))
        new_obs = np.concatenate((goal_lidar, hazards_lidar,vases_lidar, magnetometer, velo))
        
        return new_obs
