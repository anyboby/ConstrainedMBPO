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
    def termination_fn(obs, act, next_obs, safe_config):
        '''
        safeexp-pointgoal (like the other default safety-gym envs) doesn't terminate
        prematurely other than due to sampling errors etc., therefore just return Falses
        '''
        assert StaticFns.task == safe_config['task']
        obs_indices = safe_config['obs_indices']

        if safe_config['continue_goal'] == False:
            goal_dist = next_obs[...,obs_indices['goal_dist']]
            goal_met_vec = np.vectorize(StaticFns._goal_met)
            goal_met_vec.excluded.add(1)        ## exclude safe_config from vectorizing
            done = goal_met_vec(goal_dist, safe_config)
        else:
            done = np.zeros(shape=obs.shape[:-1], dtype=np.bool)
            done = done[...,None]
        return done
    
    #@anyboby  TODO this is bullshit :/
    @staticmethod
    def cost_fn(costs):
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
    def rebuild_goal(obs, act, next_obs, new_obs_pool, safe_config):
        '''
        rebuild goal, if the goal was met in the starting obs, pool of new goal_obs should be provided
        please provide unstacked observations and next_observations
        '''
        assert StaticFns.task == safe_config['task']
        obs_indices = safe_config['obs_indices']

        ### rebuild only if we already started in a goal
        goal_dist = obs[...,obs_indices['goal_dist']] 

        goal_met_vec = np.vectorize(StaticFns._goal_met)
        goal_met_vec.excluded.add(1)
        goal_met = np.squeeze(goal_met_vec(goal_dist, safe_config))

        rebuilt_obs = next_obs        
        if np.max(goal_met) == True:
            ### bit ugly but random choice in numpy is tedious
            fl_lidars = new_obs_pool[..., obs_indices['goal_lidar']]\
                .reshape(np.prod(new_obs_pool[..., obs_indices['goal_lidar']]\
                    .shape[:-1]),new_obs_pool[..., obs_indices['goal_lidar']]\
                        .shape[-1])
            fl_dist = new_obs_pool[..., obs_indices['goal_dist']]\
                .reshape(np.prod(new_obs_pool[..., obs_indices['goal_dist']]\
                    .shape[:-1]),new_obs_pool[..., obs_indices['goal_dist']]\
                        .shape[-1])
            assert fl_lidars.shape[0] == fl_dist.shape[0]
            rand_int = np.random.randint(0, fl_dist.shape[0], size=goal_met.sum())
            rebuilt_obs[goal_met, obs_indices['goal_dist']] = fl_dist[rand_int]
            rebuilt_obs[goal_met, obs_indices['goal_lidar']] = fl_lidars[rand_int]
    
        return rebuilt_obs


    @staticmethod
    def _goal_met(dist_goal, safe_config):
        assert StaticFns.task == safe_config['task']
        goal_size = safe_config['goal_size']
        ''' Return true if the current goal is met this step '''
        if 'goal' in StaticFns.task.lower():
            return dist_goal <= goal_size
        if StaticFns.task in ['x', 'z', 'circle', 'none']:
            return False
        raise ValueError(f'Invalid task {StaticFns.task}')

    @staticmethod
    def reward_np(obs, act, next_obs, safe_config):
        assert StaticFns.task == safe_config['task']
        obs_indices = safe_config['obs_indices']

        goal_dist = next_obs[...,obs_indices['goal_dist']]
        last_dist = obs[...,obs_indices['goal_dist']]

        vec_rew_fn = np.vectorize(StaticFns._reward)
        vec_rew_fn.excluded.add(2)

        reward = vec_rew_fn(goal_dist, last_dist, safe_config)
        return reward

    @staticmethod
    def _reward(dist_goal, last_dist_goal, safe_config):
        ''' Calculate the dense component of reward.  Call exactly once per step '''
        reward_distance = safe_config['reward_distance']
        reward_goal = safe_config['reward_goal']
        reward_clip = safe_config['reward_clip']
        reward_clip = 0.25 ### have to clip
    
        reward = 0.0
        # Distance from robot to goal
        if 'goal' in StaticFns.task.lower() or 'button' in StaticFns.task.lower():
            reward += (last_dist_goal - dist_goal) * reward_distance

        if reward_clip:
            in_range = reward < reward_clip and reward > -reward_clip
            if not(in_range):
                reward = np.clip(reward, -reward_clip, reward_clip)
                #print('Warning: reward was outside of range!')

        if StaticFns._goal_met(dist_goal, safe_config):
            reward += reward_goal 
        # # Distance from robot to box
        # if self.task == 'push':
        #     dist_box = self.dist_box()
        #     gate_dist_box_reward = (self.last_dist_box > self.box_null_dist * self.box_size)
        #     reward += (self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward
        #     self.last_dist_box = dist_box
        # # Distance from box to goal
        # if self.task == 'push':
        #     dist_box_goal = self.dist_box_goal()
        #     reward += (self.last_box_goal - dist_box_goal) * self.reward_box_goal
        #     self.last_box_goal = dist_box_goal
        # # Used for forward locomotion tests
        # if self.task == 'x':
        #     robot_com = self.world.robot_com()
        #     reward += (robot_com[0] - self.last_robot_com[0]) * self.reward_x
        #     self.last_robot_com = robot_com
        # # Used for jump up tests
        # if self.task == 'z':
        #     robot_com = self.world.robot_com()
        #     reward += (robot_com[2] - self.last_robot_com[2]) * self.reward_z
        #     self.last_robot_com = robot_com
        # # Circle environment reward
        # if self.task == 'circle':
        #     robot_com = self.world.robot_com()
        #     robot_vel = self.world.robot_vel()
        #     x, y, _ = robot_com
        #     u, v, _ = robot_vel
        #     radius = np.sqrt(x**2 + y**2)
        #     reward += (((-u*y + v*x)/radius)/(1 + np.abs(radius - self.circle_radius))) * self.reward_circle
        # # Intrinsic reward for uprightness
        # if self.reward_orientation:
        #     zalign = quat2zalign(self.data.get_body_xquat(self.reward_orientation_body))
        #     reward += self.reward_orientation_scale * zalign
        # Clip reward

        return reward

