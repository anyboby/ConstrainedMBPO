from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .base_sampler import BaseSampler
ACTION_PROCESS_ENVS = [
    'Safexp-PointGoal2',
    ]

class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._last_action = None
        self.process_act_vec = np.vectorize(self.process_act)    ###vectorize elementwise function process_act to work for np arrays

    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              terminal,
                              next_observation,
                              info):

        if self._obs_process_type in ACTION_PROCESS_ENVS:
            #### concatenate act, last act and an acc_spike prediction signal based on these
            action_proc = np.concatenate((action, self._last_action, np.array([self.process_act(action[0], self._last_action[0])])))
        else:
            action_proc = action

        processed_observation = {
            'observations': observation,
            'actions': action_proc,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def process_act(self, act, last_act):
        '''
        Predicts a spike based on 0-transition between actions
        !! very specifically designed for x-acceleration spike detection
        returns a normalized prediction signal for y-acceleration in mujoco envs
        a shape (1,) np array
        
        '''
        act_x = act
        last_act_x = last_act
        acc_spike = 0
        ### acc
        if last_act_x==act_x:
            acc_spike=0
        else:
            if last_act_x<=0<=act_x or act_x<=0<=last_act_x:
                #pass
                acc_spike = act_x-last_act_x
                acc_spike = acc_spike/abs(acc_spike) #normalize
        return acc_spike



    def sample(self):
        if self._current_observation is None:
            self._current_observation = np.squeeze(self.env.reset())
            self._last_action = np.zeros(shape=self.env.action_space.shape)

        action = self.policy.actions_np([
            self.env.convert_to_active_observation(
                self._current_observation)[None]
        ])[0]

        next_observation, reward, terminal, info = self.env.step(action)
        next_observation = np.squeeze(next_observation)
        reward = np.squeeze(reward)
        terminal = np.squeeze(terminal)
        info = info.get(0, {})      ## @anyboby not very clean, only works for 1 env in parallel

        # just for testing
        #self.env.render()
        
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        #### add to pool only after full epoch or terminal path
        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            
            self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)
            self._last_action = np.zeros(shape=self.env.action_space.shape)
            self._n_episodes += 1
        else:
            self._current_observation = next_observation
            self._last_action = action

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics
