"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

import numpy as np
import gym
from gym import spaces, wrappers
import safety_gym
from safety_gym.envs.engine import Engine
import json
import os

from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym import register_environments
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from softlearning.environments.adapters.safety_preprocessed_wrapper import SafetyPreprocessedEnv
from collections import defaultdict
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.vec_env import DummyVecEnv


def parse_domain_task(gym_id):
    domain_task_parts = gym_id.split('-')
    domain = '-'.join(domain_task_parts[:1])
    task = '-'.join(domain_task_parts[1:])

    return domain, task


CUSTOM_GYM_ENVIRONMENT_IDS = register_environments()
CUSTOM_GYM_ENVIRONMENTS = defaultdict(list)

for gym_id in CUSTOM_GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    CUSTOM_GYM_ENVIRONMENTS[domain].append(task)

CUSTOM_GYM_ENVIRONMENTS = dict(CUSTOM_GYM_ENVIRONMENTS)

GYM_ENVIRONMENT_IDS = tuple(gym.envs.registry.env_specs.keys())
GYM_ENVIRONMENTS = defaultdict(list)


for gym_id in GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    GYM_ENVIRONMENTS[domain].append(task)

GYM_ENVIRONMENTS = dict(GYM_ENVIRONMENTS)

SAFETY_WRAPPER_IDS = {
    'Safexp-PointGoal2-v0':SafetyPreprocessedEnv,
    'Safexp-PointGoal1-v0':SafetyPreprocessedEnv,
    'Safexp-PointGoal0-v0':SafetyPreprocessedEnv,
}

class GymAdapter(SoftlearningEnv):
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
        super(GymAdapter, self).__init__(domain, task, *args, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env_id = f"{domain}-{task}"
            env = gym.envs.make(env_id, **kwargs)

            #env_id = f""
            #env = gym.make("Safexp-PointGoal1-v0")
            
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
            env = NormalizeActionWrapper(env)


        #### --- specifically for safety_gym wrappring --- ###
        if env_id in SAFETY_WRAPPER_IDS:
            dirname, _ = os.path.split(os.path.abspath(__file__))
            #### load config file
            with open(f'{dirname}/../gym/safety_gym/configs/{env_id}_config.json', 'r') as fp:
                self.safeconfig = json.load(fp)
            fp.close()
            with open(f'{dirname}/../gym/safety_gym/add_configs/{env_id}_add_config.json', 'r') as fp:
                add_config = json.load(fp)
            fp.close()


            env = Engine(self.safeconfig)
            env = SAFETY_WRAPPER_IDS[env_id](env)

            #### additional config info like stacking etc.
            for k in add_config.keys():
                self.safeconfig[k] = add_config[k]
                    
            #### dump config file to current data dir
            with open(f'{env_id}_config.json', 'w') as fp:
                json.dump(self.safeconfig, fp)
            fp.close()
            ####

            ### adding unserializable additional info after dumping (lol)
            self.obs_indices = env.obs_indices
            self.safeconfig['obs_indices'] = self.obs_indices
            #### check if extended action space exists:
            if hasattr(env, 'action_space_ext'):
                self.action_space_ext = env.action_space_ext
                self.safeconfig['action_space_ext'] = self.action_space_ext

            ### stack env
            self.stacks = self.safeconfig['stacks'] ### for convenience
            self.stacking_axis = self.safeconfig['stacking_axis']
            env = DummyVecEnv([lambda:env])
            #env = VecNormalize(env)        doesn't work at all for some reason
            env = VecFrameStack(env, self.safeconfig['stacks'])

        #### --- end specifically for safety_gym  --- ###


        self._env = env

    @property
    def observation_space(self):
        observation_space = self._env.observation_space
        return observation_space

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
            return super(GymAdapter, self).active_observation_shape

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
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        return action_space

    def step(self, action, *args, **kwargs):
        # TODO(hartikainen): refactor this to always return an OrderedDict,
        # such that the observations for all the envs is consistent. Right now
        # some of the gym envs return np.array whereas others return dict.
        #
        # Something like:
        # observation = OrderedDict()
        # observation['observation'] = env.step(action, *args, **kwargs)
        # return observation

        if self.stacks>1:       ### because VecEnv has additional dim for parallel envs
            action=np.array([action])
            #action=np.array([[0.0,-0.4]])
        return self._env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

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
    def unwrapped(self):
        return self._env.unwrapped

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError
