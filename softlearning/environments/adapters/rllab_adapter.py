"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

import numpy as np
import gym
from gym import spaces, wrappers
from gym.spaces import Box, Discrete

import json
import os
import sys
import softlearning.environments.rllab as rllab_dir
path = os.path.dirname(rllab_dir.__file__)
sys.path.append(path)

from rllab.spaces.box import Box as rllBox
from rllab.spaces.discrete import Discrete as rllDiscrete
from rllab.spaces.product import Product as rllProduct

from softlearning.environments.rllab.rllab.envs.normalized_env import NormalizedEnv
from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym import register_environments
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from softlearning.environments.adapters.safety_preprocessed_wrapper import SafetyPreprocessedEnv
from collections import defaultdict
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.vec_env import DummyVecEnv

from sandbox.cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv
from sandbox.cpo.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from sandbox.cpo.envs.mujoco_safe.ant_env_safe import SafeAntEnv
from sandbox.cpo.envs.mujoco_safe.point_env_safe import SafePointEnv
from sandbox.cpo.envs.mujoco_safe.half_cheetah_env_safe import SafeHalfCheetahEnv
from sandbox.cpo.envs.mujoco_safe.simple_humanoid_env_safe import SafeSimpleHumanoidEnv

def parse_domain_task(gym_id):
    domain_task_parts = gym_id.split('-')
    domain = '-'.join(domain_task_parts[:1])
    task = '-'.join(domain_task_parts[1:])

    return domain, task
RLLAB_ENVIRONMENTS = {
    'PointGather':['v0'],
    'AntGather':['v0'],
    'AntCircle':['v0'],
    'PointCircle':['v0'],
    'HalfCheetahCircle':['v0'],
    'SimpleHumanoidCircle':['v0'],
}

OVERWRITE = {
    'SimpleHumanoidCircle-v0':{
        'observation_space': lambda x: Box(low=np.ones(shape=36)*x.low[0], high=np.ones(shape=36)*x.high[0]),
        'action_space': lambda x: Box(low=np.ones(shape=10)*x.low[0], high=np.ones(shape=10)*x.high[0]),
        'next_observation': lambda x: np.delete(x, np.s_[-69:-3]),
    }
}

RLLAB_ENTRIES = {
    'PointGather-v0':PointGatherEnv,
    'AntGather-v0':AntGatherEnv,
    'AntCircle-v0':SafeAntEnv,
    'PointCircle-v0':SafePointEnv,
    'HalfCheetahCircle-v0':SafeHalfCheetahEnv,
    'SimpleHumanoidCircle-v0':SafeSimpleHumanoidEnv,
}
RLLAB_KWARGS = {
    'PointGather-v0':{
        'apple_reward':10,
        'bomb_cost':1,
        'n_apples':2,
        'activity_range':6,
        },
    'AntGather-v0':{
        'apple_reward':10,
        'bomb_cost':1,
        'n_apples':2,
        'activity_range':6,
        },
    'AntCircle-v0':{                       
        'xlim':3,
        'circle_mode':True,
        'target_dist':10,
        'abs_lim':True,
        },
    'PointCircle-v0':{},
    'HalfCheetahCircle-v0':{},
    'SimpleHumanoidCircle-v0':{
        'xlim':2.5,
        'circle_mode':True,
        'target_dist':10,
        'abs_lim':True,
    },
}

#### Cost params according to cpo paper
RLLAB_COST_PARAMS = {
    'AntCircle-v0': {
        'cidx':-3,
        'xlim': 3,
        'target_dist': 10 },
    'SimpleHumanoidCircle-v0':{
        'cidx':-3,
        'xlim': 2.5,
        'target_dist': 10 },
    }

def eval_cost_gather(obs, info, env_id):
    return info['bombs']

def eval_cost_mjc(obs, info, env_id):
    return float(np.abs(obs[RLLAB_COST_PARAMS[env_id]['cidx']]) >= RLLAB_COST_PARAMS[env_id]['xlim'])

RLLAB_COSTF = {
    'PointGather-v0':eval_cost_gather,
    'AntGather-v0':eval_cost_gather,
    'AntCircle-v0':eval_cost_mjc,
    'PointCircle-v0':eval_cost_mjc,
    'HalfCheetahCircle-v0':eval_cost_mjc,
    'SimpleHumanoidCircle-v0':eval_cost_mjc,
}

class RllabAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Gym envs."""

    def __init__(self,
                 domain,
                 task,
                 *args,
                 env=None,
                 normalize=True,
                 observation_keys=None,
                 unwrap_time_limit=True,
                 **kwargs):
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")

        self.normalize = normalize
        self.observation_keys = observation_keys
        self.unwrap_time_limit = unwrap_time_limit
        self.stacks = 1
        self.stacking_axis = 0

        self._Serializable__initialize(locals())
        super(RllabAdapter, self).__init__(domain, task, *args, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            self.env_id = env_id = f"{domain}-{task}"
            env = RLLAB_ENTRIES[env_id](**RLLAB_KWARGS[env_id])
            
        else:
            assert domain is None and task is None, (domain, task)
            env_id = 'custom'

        if isinstance(env, wrappers.TimeLimit) and unwrap_time_limit:
            # Remove the TimeLimit wrapper that sets 'done = True' when
            # the time limit specified for each environment has been passed and
            # therefore the environment is not Markovian (terminal condition
            # depends on time rather than state).
            env = env.env

        if isinstance(env.observation_space, spaces.Dict):
            observation_keys = (
                observation_keys or list(env.observation_space.spaces.keys()))
        if normalize:
            env = NormalizedEnv(env)

        self._env = env

    @property
    def observation_space(self):
        observation_space = self._env.observation_space
        obs_space_gym = self.convert_to_gym_space(observation_space)
        if self.env_id in OVERWRITE:
            obs_space_gym = OVERWRITE[self.env_id]['observation_space'](obs_space_gym)        
        return obs_space_gym

    @property
    def active_observation_shape(self):
        """Shape for the active observation based on observation_keys. 
        returns latest observation in case of a stacked observation."""
        if self.stacks>1:
            active_size = list(self._env.observation_space.shape)
            active_size[self.stacking_axis] = int(self._env.observation_space.shape[self.stacking_axis]/self.stacks)
            active_size = tuple(active_size)
            return active_size

        if not isinstance(self._env.observation_space, spaces.Dict):
            return super(RllabAdapter, self).active_observation_shape

        observation_keys = (
            self.observation_keys
            or list(self._env.observation_space.spaces.keys()))

        active_size = sum(
            np.prod(self._env.observation_space.spaces[key].shape)
            for key in observation_keys)

        active_observation_shape = (active_size, )

        return active_observation_shape

    def convert_to_active_observation(self, observation):
        if self.stacks>1:
            old_obs_len = self._env.observation_space.shape[self.stacking_axis] - int(self._env.observation_space.shape[self.stacking_axis]/self.stacks)
            active_obs = np.delete(observation, slice(old_obs_len), self.stacking_axis)
            return active_obs

        if not isinstance(self._env.observation_space, spaces.Dict):
            return observation

        observation_keys = (
            self.observation_keys
            or list(self._env.observation_space.spaces.keys()))

        observation = np.concatenate([
            observation[key] for key in observation_keys
        ], axis=-1)

        return observation

    @property
    def action_space(self, *args, **kwargs):
        action_space = self._env.action_space
        action_space_gym = self.convert_to_gym_space(action_space)
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        if self.env_id in OVERWRITE:
            action_space_gym = OVERWRITE[self.env_id]['action_space'](action_space_gym)
        return action_space_gym

    def step(self, action, *args, **kwargs):
        if isinstance(self._env, VecFrameStack):       ### because VecEnv has additional dim for parallel envs
            action=np.array([action])
        o, r, done, info = self._env.step(action, *args, **kwargs)
        info['cost'] = RLLAB_COSTF[self.env_id](o, info, self.env_id)
        
        if self.env_id in OVERWRITE:
            o = OVERWRITE[self.env_id]['next_observation'](o)

        return o, r, done, info

    def reset(self, *args, **kwargs):
        o = self._env.reset(*args, **kwargs)
        if self.env_id in OVERWRITE:
            o = OVERWRITE[self.env_id]['next_observation'](o)
        return o

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def get_sim_state(self, *args, **kwargs):       #### @anyboby
        assert hasattr(self._env, 'get_sim_state')
        return self._env.get_sim_state(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    @property
    def domain(self):
        return self._domain

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def convert_to_gym_space(self, space):
        if isinstance(space, rllBox):
            return Box(low=space.low, high=space.high)
        elif isinstance(space, rllDiscrete):
            return Discrete(n=space.n)
        else:
            raise NotImplementedError
