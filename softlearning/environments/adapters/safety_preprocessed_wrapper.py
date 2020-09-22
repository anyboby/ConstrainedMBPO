import numpy as np 
from scipy import signal
import gym

import matplotlib.pyplot as plt
from math import log as log
from math import exp as e
import math


class SafetyPreprocessedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(SafetyPreprocessedEnv, self).__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, (env.robot.nu,), dtype=np.float32) 
        # self.b, self.a = signal.butter(3, 0.1)
        # self.obs_replay_vy_real = []
        # self.obs_replay_vy_filt = []
        # self.obs_replay_acc_y_real = []
        # self.obs_replay_acc_y_filt = []
        
        self.prev_obs = None
        self.prev_act = np.zeros(self.action_space.shape)

        # ## inits for action processing
        # self.prev_acc_spike = 0
        # self.time_since_spike = 0
        # self.transform_a_vec = np.vectorize(transform_a)
        # self.spike_vec = np.vectorize(spike)

        # ###@anyboby testing
        # self._cum_cost = 1e-8

        # self.remove_obs = [
        #     'accelerometer',
        #     'gyro',
        #     #'ctrl',
        # ]
        # self.add_obs = 3
        # self.obs_flat_size = sum([np.prod(i.shape) for i in self.env.obs_space_dict.values()])+self.add_obs
        # self.obs_flat_size = self.obs_flat_size-sum([np.prod(self.env.obs_space_dict[i].shape) for i in self.remove_obs])
        # self.observation_space = gym.spaces.Box(-np.inf, np.inf, ((self.obs_flat_size),), dtype=np.float32)  #manually set size, add. dim for ctrl spike

        self.obs_flat_size = sum([np.prod(i.shape) for i in self.env.obs_space_dict.values()])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, ((self.obs_flat_size),), dtype=np.float32)
        self.obs_indices = {}
        k_size = 0
        offset = 0
        for k in sorted(self.env.obs_space_dict.keys()):
            # if k not in self.remove_obs:
            #     k_size = np.prod(self.env.obs_space_dict[k].shape)
            #     self.obs_indices[k] = slice(offset,offset + k_size)
            #     offset += k_size
            k_size = np.prod(self.env.obs_space_dict[k].shape)
            self.obs_indices[k] = slice(offset,offset + k_size)
            offset += k_size


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        #self.obs_replay_vy = [observation[58]]
        observation = self.preprocess_obs(observation, action=np.zeros(self.action_space.shape))
        self.prev_obs = observation
        #stacked_obs = np.concatenate((observation, self.prev_obs))
        return self.observation(observation)

    def step(self, action):
        # a_transformed = action #self.transform_a_vec(action)
        observation, reward, done, info = self.env.step(action)
        self.prev_obs_unprocessed = observation
        observation = self.preprocess_obs(observation, action)
        #stacked_obs = np.concatenate((observation, self.prev_obs))
        self.prev_obs = observation
        self.prev_act = action
        return self.observation(observation), reward, done, info

    def observation(self, observation):

        return observation


    def preprocess_obs(self, obs, action):
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
            # if key not in self.remove_obs:
            #     flat_obs[index] = obs[key].flat
            # #### replace goal_dist with true dist
            # if key == 'goal_dist':
            #     flat_obs[index] = self.env.dist_goal()
            flat_obs[index] = obs[key].flat
        obs = flat_obs

        # pos = self.env.world.robot_pos()
        # vel = self.env.world.robot_vel()

        # obs[-3:] = self.env.world.robot_pos()
        #obs[-1] = self._cum_cost/self.cost_lim

        #self.obs_indices['pos']=slice(self.obs_flat_size-3,self.obs_flat_size)
        
        #additional obs
        #flat_obs[-1]=spike_2(action[0], self.prev_act[0])
        return obs

def transform_a(ax):
    e_x = 0.8             # edge_x
    e_y = 0.008            # edge_y
    c_1 = (1-e_y)/(1-e_x) # steepness before edge
    c_2 = e_y/e_x         # steepness after edge
    e_s = 15              # smoothness
    a = c_1*ax+1/e_s*log(e(e_s*(ax+e_x))+1)*(c_2-c_1)+ \
            1/e_s*log(e(e_s*e_x)+e(e_s*ax))*(c_1-c_2)
    return a

def _delta(x):
    a = 0.03
    return 2/math.sqrt(math.pi)*e(-((x-0.05)/a)**8)-2/math.sqrt(math.pi)*e(-((x+0.05)/a)**8)
def spike(x1,x2):
    if x1>=x2:    
        delta = _delta(x1)-_delta(x2)
    elif x1<x2:
        delta = _delta(x2)-_delta(x1)
    elif ((x1>0.05)and(x2>0.05)):
        delta = _delta(0.05)-delta(x2)
    elif ((x1<0.05)and(x2<0.05)):
        delta = _delta(0.05)-_delta(x2)
    delta = np.clip(delta, -2, 2)
    return delta

def spike_2(x1, x2):
    '''
    Predicts a spike based on 0-transition between actions
    !! very specifically designed for x-acceleration spike detection
    returns a normalized prediction signal for y-acceleration in mujoco envs
    a shape (1,) np array
    
    '''
    acc_spike = 0
    ### acc
    if x1==x2:
        acc_spike=0
    else:
        if x2<=0<=x1 or x1<=0<=x2:
            #pass
            acc_spike = x1-x2
            acc_spike = acc_spike/abs(acc_spike) #normalize
    return acc_spike
