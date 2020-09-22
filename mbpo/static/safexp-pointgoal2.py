import numpy as np
import sys, os
import json

class StaticFns:
    task = 'Safexp-PointGoal2-v1'

        # goal_size = config.goal_size
        # reward_distance = config.reward_distance
        # reward_clip = config.reward_clip
        # reward_goal = config.reward_goal

    @staticmethod
    def termination_fn(obs, act, next_obs, env):
        '''
        safeexp-pointgoal (like the other default safety-gym envs) doesn't terminate
        prematurely other than due to sampling errors etc., therefore just return Falses
        '''
        
        obs_space_dict = env.unwrapped.obs_space_dict
        config = env.unwrapped.config

        if config['continue_goal'] == False:
            gd_slice = StaticFns.obs_indices('goal_dist', obs_space_dict)
            goal_dist = -np.log(np.clip(next_obs[...,gd_slice], 1e-3, 1e3))
            goal_met_vec = np.vectorize(StaticFns._goal_met)
            goal_met_vec.excluded.add(1)   
            done = goal_met_vec(goal_dist)
        else:
            done = np.zeros(shape=obs.shape[:-1], dtype=np.bool)
            done = done[...,None]
        return done
    
    #@anyboby  TODO @anyboby :/
    @staticmethod
    def cost_fn(costs, env):
        """
        interprets model ensemble regression output as probability and converts to binary cost between 1 and 0
        Args:
            costs: expects samples along axis 0
        """

        #clip to 0 and 1
        cost_clipped = np.clip(costs, 0, 1)
        cost_rounded = np.around(cost_clipped)
        #cost_rounded = cost_clipped
    
        return cost_rounded
        #

    
    @staticmethod
    def rebuild_goal(obs, act, next_obs, new_obs_pool, env):
        '''
        rebuild goal, if the goal was met in the starting obs, pool of new goal_obs should be provided
        please provide unstacked observations and next_observations
        '''
        obs_space_dict = env.unwrapped.obs_space_dict
        config = env.unwrapped.config
        gd_slice = StaticFns.obs_indices('goal_dist', obs_space_dict)
        gl_slice = StaticFns.obs_indices('goal_lidar', obs_space_dict)

        ### rebuild only if we already started in a goal
        goal_dist = -np.log(np.clip(next_obs[...,gd_slice], 1e-3, 1e3))
        goal_size = config['goal_size']
        
        ## seems a bit undirect
        goal_met = goal_dist <= goal_size
        goals_met = np.sum(goal_met)


        rebuilt_obs = np.array(next_obs)
        if goals_met > 0:
            goal_met = np.repeat(goal_met, repeats=obs.shape[-1], axis=-1)
            goal_ind_mask = np.zeros_like(goal_met, dtype='bool')
            goal_ind_mask[...,gd_slice] = True
            goal_ind_mask[...,gl_slice] = True
            goal_met = np.logical_and(goal_met, goal_ind_mask)

            ### bit ugly but random choice in numpy is tedious
            fl_lidars = new_obs_pool[..., gl_slice]\
                .reshape(np.prod(new_obs_pool[..., gl_slice].shape[:-1]),\
                    new_obs_pool[..., gl_slice].shape[-1])
            fl_dist = new_obs_pool[..., gd_slice]\
                .reshape(np.prod(new_obs_pool[..., gd_slice]\
                    .shape[:-1]),new_obs_pool[..., gd_slice]\
                        .shape[-1])
            assert fl_lidars.shape[0] == fl_dist.shape[0]
            fl_obs_pool = np.concatenate((fl_dist, fl_lidars), axis=-1)
            
            rand_mask = np.zeros_like(fl_obs_pool, dtype='bool')
            rand_ind = np.random.choice(rand_mask.shape[0], size=goals_met, replace=False)
            rand_mask[rand_ind,:] = True

            rebuilt_obs[goal_met] = fl_obs_pool[rand_mask]
    
        return rebuilt_obs

    @staticmethod
    def _goal_met(dist_goal, config):
        goal_size = config['goal_size']
        ''' Return true if the current goal is met this step '''
        if 'goal' in StaticFns.task.lower():
            return dist_goal <= goal_size
        if StaticFns.task in ['x', 'z', 'circle', 'none']:
            return False
        raise ValueError(f'Invalid task {StaticFns.task}')

    @staticmethod
    def reward_f(obs, act, next_obs, env):
        obs_space_dict = env.unwrapped.obs_space_dict
        config = env.unwrapped.config
        gd_slice = StaticFns.obs_indices('goal_dist', obs_space_dict)
        gl_slice = StaticFns.obs_indices('goal_lidar', obs_space_dict)

        reward_distance = config['reward_distance']
        reward_goal = config['reward_goal']
        goal_size = config['goal_size']

        goal_dist = -np.log(np.clip(next_obs[...,gd_slice], 1e-3, 1e3))
        last_dist = -np.log(np.clip(obs[...,gd_slice], 1e-3, 1e3))

        ### Calculate the dense component of reward.  Call exactly once per step
        reward = np.zeros(shape=act.shape[:-1])[...,None]
        # Distance from robot to goal
        if 'goal' in StaticFns.task.lower() or 'button' in StaticFns.task.lower():
            reward += (last_dist - goal_dist) * reward_distance

        # if reward_clip:
        #     # in_range = reward < reward_clip and reward > -reward_clip
        #     # if not(in_range):
        #     reward = np.clip(reward, -reward_clip, reward_clip)
        
        ### goal rew
        ### Return true if the current goal is met this step 
        if 'goal' in StaticFns.task.lower():
            reward[goal_dist <= goal_size] += reward_goal
        
        return reward

    @staticmethod
    def prior_f(obs, acts):
        '''
        Predicts a spike based on 0-transition between actions
        !! very specifically designed for acceleration spike detection
        returns a normalized prediction signal for y-acceleration in mujoco envs
        a shape (1,) np array
        
        '''
        last_acts = obs[...,:2]
        acc_spikes = np.zeros_like(last_acts)

        ### mask where sign of acc changes or either of the accelerations drops to zero
        sign_mask = np.logical_or(np.logical_or(last_acts/(acts+1e-8)<0, abs(last_acts/(acts+10))<1e-8), abs(acts/(last_acts+10))<1e-8)
        acc_spikes[sign_mask] = (acts[sign_mask]-last_acts[sign_mask])/(abs((acts[sign_mask]-last_acts[sign_mask]))+1e-8)
        
        return acc_spikes
    
    @staticmethod
    def post_f(obs, acts):
        assert len(obs.shape) == len(acts.shape)
        if len(obs.shape)==1:
            obs = obs[None]
            acts = acts[None]
            return_single = True
        else: return_single = False

        obs[...,:2] = acts[...,:2]
        if return_single:
            return obs[0] 
        else: 
            return obs
    
    @staticmethod
    def obs_indices(key, obs_space_dict):
        ind = 0
        for k in sorted(obs_space_dict):
            if k == key:
                return slice(ind, ind + np.prod(obs_space_dict[k].shape))
            ind += np.prod(obs_space_dict[k].shape)
        
        raise ValueError('Key not found in obs dict')
