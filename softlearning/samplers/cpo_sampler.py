from collections import defaultdict
from collections import deque, OrderedDict
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt

from softlearning.policies.safe_utils.logx import EpochLogger
from softlearning.policies.safe_utils.mpi_tools import mpi_sum

from .base_sampler import BaseSampler
ACTION_PROCESS_ENVS = [
    'Safexp-PointGoal2',
    ]

class CpoSampler():
    def __init__(self,
                 max_path_length,
                 min_pool_size,
                 batch_size,
                 store_last_n_paths = 10,
                 preprocess_type='default',
                 logger = None):
        self._max_path_length = max_path_length
        self._path_length = 0
        self._path_return = 0
        self._path_cost = 0
        self.cum_cost = 0
        
        if logger:
            self.logger = logger
        else: 
            self.logger = EpochLogger()

        self._store_last_n_paths = store_last_n_paths
        self._last_n_paths = deque(maxlen=store_last_n_paths)

        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._last_action = None
        
        #### cost options        
        self.cum_cost = 0
        self.cost_lim_at1000 = 50
        gamma = 0.99
        ep_len = 1000
        self.cost_lim = self.cost_lim_at1000/ep_len*(1-gamma**(ep_len+1))/(1-gamma)
        self.penalty_coeff = 5
        self.penalty_lr = 0.0005
        self.penalty_clip = 0.01
        #self.penalty_dic = 0.99
        self.learn_penalty = False
        self.learn_penalty_mult = False



        self._obs_process_type = preprocess_type
        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.vf_ensemble_size = self.policy.vf_ensemble_size
        self.pool = pool
        self.vf_is_gaussian = self.policy.vf_is_gaussian

    def set_policy(self, policy):
        self.policy = policy

    def set_logger(self, logger):
        """
        provide a logger (Sampler creates it's own logger by default, 
        but you might want to share a logger between algo, samplers, etc.)
        
        automatically shares logger with agent
        Args: 
            logger : instance of EpochLogger
        """ 
        self.logger = logger        

    def terminate(self):
        self.env.close()

    def get_diagnostics(self):
        diagnostics = OrderedDict({'pool-size': self.pool.size})
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics

    def __getstate__(self):
        state = {
            key: value for key, value in self.__dict__.items()
            if key not in ('env', 'policy', 'pool')
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.env = None
        self.policy = None
        self.pool = None

    def clear_last_n_paths(self):
        self._last_n_paths.clear()

    @property
    def max_path_length(self):
        return self._max_path_length

    def get_last_n_paths(self, n=None):
        if n is None:
            n = self._store_last_n_paths

        last_n_paths = tuple(islice(self._last_n_paths, None, n))

        return last_n_paths

    def batch_ready(self):
        return self.pool.size >= self.pool.max_size

    def penalty_mult(self, cur_cost):
        penalty_mult = 0.99**(self.penalty_coeff*cur_cost)
        return penalty_mult

    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              cost,
                              terminal,
                              next_observation,
                              info):

        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'cost'   : [cost],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def sample(self, timestep):
        if self._current_observation is None:
            # Reset environment
            self._current_observation, reward, terminal, c = np.squeeze(self.env.reset()), 0, False, 0
            self._last_action = np.zeros(shape=self.env.action_space.shape)

        # Get outputs from policy
        # test_obs = [self._current_observation[np.newaxis] for i in range(100)]
        # test_obs = np.concatenate(test_obs, axis=0)
        get_action_outs = self.policy.get_action_outs(self._current_observation, factored=True, inc_var=True)
        #get_action_outs = self.policy.get_action_outs(self._current_observation)

        a = get_action_outs['pi']
        v_t = get_action_outs['v'][:,0]
        vc_t = get_action_outs['vc'][:,0]  # Agent may not use cost value func
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']

        v_var = get_action_outs['v_var'][:,0]
        vc_var = get_action_outs['vc_var'][:,0]

        next_observation, reward, terminal, info = self.env.step(a)
        next_observation = np.squeeze(next_observation)
        reward = np.squeeze(reward)
        terminal = np.squeeze(terminal)
        # info = info[0]      ## @anyboby not very clean, only works for 1 env in parallel
        
        c = info.get('cost', 0)
        self.cum_cost += c


        #### @anyboby do some testing
        if self.learn_penalty:
            reward = reward - self.penalty_coeff*c
        if self.learn_penalty_mult:
            reward = reward * self.penalty_mult(self.cum_cost)

        #save and log
        self.pool.store(self._current_observation, a, next_observation, reward, v_t, v_var, c, vc_t, vc_var, logp_t, pi_info_t, terminal, timestep)
        self.logger.store(VVals=v_t, CostVVals=vc_t, VVars = v_var, CostVVars=vc_var)
        
        self._path_length += 1
        self._path_return += reward
        self._path_cost += c
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=a,
            reward=reward,
            cost=c,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)


        #### update current obs before finishing
        self._current_observation = next_observation
        self._last_action = a

        #### add to pool only after full epoch or terminal path
        if terminal or self._path_length >= self._max_path_length:

            # If trajectory didn't reach terminal state, bootstrap value target(s)
            if terminal and not(self._path_length >= self._max_path_length):
                # Note: we do not count env time out as true terminal state,
                ## But costs are calculated for the maximum episode length, 
                ## even for early termination
                
                self.finish_all_paths(append_val=False, append_cval=True)
            else:
                self.finish_all_paths(append_val=True, append_cval=True)
            
        return next_observation, reward, terminal, info


    def finish_all_paths(self, append_val=False, append_cval=False, reset_path = True):
            if self._current_observation is None:   #return if already finished
                return

            ####--------------------####
            ####  finish pool traj  ####
            ####--------------------####
            # If trajectory didn't reach terminal state, bootstrap value target(s)
            if not append_val:
                # Note: we do not count env time out as true terminal state
                last_val, last_val_var = np.zeros((self.vf_ensemble_size,1)), \
                                            np.zeros((self.vf_ensemble_size,1))
                                                                    
            else:
                last_val, last_val_var = self.policy.get_v(self._current_observation, factored=True, inc_var=True)
            
            if not append_cval:
                last_cval, last_cval_var = np.zeros((self.vf_ensemble_size,1)), \
                                                np.zeros((self.vf_ensemble_size,1))
            else:
                last_cval, last_cval_var = self.policy.get_vc(self._current_observation, factored=True, inc_var=True)
            
            self.pool.finish_path(last_val, last_val_var, last_cval, last_cval_var)


            ####--------------------####
            ####  finish path       ####
            ####--------------------####

            if reset_path:
                self.logger.store(RetEp=self._path_return, EpLen=self._path_length, CostEp=self._path_cost, CostFullEp=self._path_cost/self._path_length * self._max_path_length)
                self.last_path = {
                    field_name: np.array(values)
                    for field_name, values in self._current_path.items()
                }
                self._last_n_paths.appendleft(self.last_path)

                self._max_path_return = max(self._max_path_return,
                                            self._path_return)
                self._last_path_return = self._path_return

                self.policy.reset() #does nohing for cpo policy atm
                self._current_observation = None
                self._last_action = np.zeros(shape=self.env.action_space.shape)
                self._path_length = 0
                self._path_return = 0
                self._path_cost = 0
                self._current_path = defaultdict(list)
                self._n_episodes += 1

            #### adjust penalty if needed
            if self.learn_penalty and self.batch_ready():
                cur_cost_margin = self.logger.get_stats('CostEp')[0]-self.cost_lim_at1000
                old_penalty_coeff = self.penalty_coeff
                new_penalty_coeff = self.penalty_lr * cur_cost_margin
                #self.penalty_coeff += np.clip(new_penalty_coeff-old_penalty_coeff, -self.penalty_clip, self.penalty_clip)
                self.penalty_coeff += new_penalty_coeff
                self.penalty_coeff = max(self.penalty_coeff, 0)
                print(f'new penalty coeff: {self.penalty_coeff}')

            if self.learn_penalty_mult and self.batch_ready():
                cur_cost_margin = self.logger.get_stats('CostEp')[0]-self.cost_lim_at1000
                self.penalty_coeff += self.penalty_lr * cur_cost_margin
                print(f'new penalty coeff: {self.penalty_coeff}')



    def log(self):
        """
        logs several stats over the timesteps since the last 
        flush (such as epCost, totalCost etc.)
        """
        logger = self.logger
        cumulative_cost = mpi_sum(self.cum_cost)    
        cost_rate = cumulative_cost / self._total_samples

        # Performance stats
        logger.log_tabular('RetEp', with_min_and_max=True)
        logger.log_tabular('CostEp', with_min_and_max=True)
        logger.log_tabular('CostFullEp', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CostCumulative', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)

        # Value function values
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('CostVVals', with_min_and_max=True)

        # Time and steps elapsed
        # logger.log_tabular('TotalEnvInteracts', self._total_samples)
        