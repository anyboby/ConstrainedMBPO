from collections import defaultdict
from collections import deque, OrderedDict
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
import random

from softlearning.samplers.cpo_sampler import CpoSampler
from softlearning.policies.safe_utils.logx import EpochLogger
from softlearning.policies.safe_utils.mpi_tools import mpi_sum

from .base_sampler import BaseSampler
ACTION_PROCESS_ENVS = [
    'Safexp-PointGoal2',
    ]

class ModelSampler(CpoSampler):
    def __init__(self,
                 max_path_length,
                 min_pool_size,
                 batch_size=1000,
                 store_last_n_paths = 10,
                 preprocess_type='default',
                 logger = None):
        self._max_path_length = max_path_length
        self._path_length = np.zeros(batch_size)
        self._path_return = np.zeros(batch_size)
        self._path_cost = np.zeros(batch_size)

        self._alive_paths = np.ones(batch_size, dtype=np.bool)

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
        self.cum_cost = 0

        self.batch_size = batch_size

        self._obs_process_type = preprocess_type
        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_debug_buf(self, pool):
        self.pool_debug = pool

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


    def get_last_n_paths(self, n=None):
        if n is None:
            n = self._store_last_n_paths

        last_n_paths = tuple(islice(self._last_n_paths, None, n))

        return last_n_paths

    def batch_ready(self):
        return self.pool.size >= self.pool.max_size

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
            'rewards': reward,
            'cost'   : cost,
            'terminals': terminal,
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation


    def sample(self):
        if self._current_observation is None:
            # Reset environment
            self._current_observation, reward, terminal, c = \
                        np.squeeze(self.env.reset()), \
                        np.zeros(self.batch_size), \
                        np.zeros(self.batch_size, dtype=np.bool), \
                        np.zeros(self.batch_size)

        self._n_episodes += 1

        batch_size = self.batch_size
        # Get outputs from policy
        test_obs = [self._current_observation[np.newaxis] for i in range(batch_size)]
        test_obs = np.concatenate(test_obs, axis=0)
        get_action_outs = self.policy.get_action_outs(test_obs)
        #get_action_outs = self.policy.get_action_outs(self._current_observation)

        a = get_action_outs['pi']
        v_t = get_action_outs['v']
        vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']

        next_observation, reward, terminal, info = self.env.step(a[0])

        next_observation = np.squeeze(next_observation)
        test_nextobs = np.squeeze(next_observation)
        test_nextobs = [test_nextobs[np.newaxis] for i in range(batch_size)]
        test_nextobs = np.concatenate(test_nextobs, axis=0)

        test_rew = np.squeeze(reward)
        test_rew = [test_rew[np.newaxis] for i in range(batch_size)]
        test_rew = np.concatenate(test_rew, axis=0)

        test_term = np.squeeze(terminal)
        test_term = [test_term[np.newaxis] for i in range(batch_size)]
        test_term = np.concatenate(test_term, axis=0)
        random_terms = 50000*np.random.rand(*test_term[self._alive_paths].shape)>49999
        test_term[self._alive_paths] += random_terms

        #info = info[0]      ## @anyboby not very clean, only works for 1 env in parallel
        #info = np.squeeze(info)
        test_info = [info for i in range(batch_size)]
        test_info= np.concatenate(test_info, axis=0)
        
        c = info[0].get('cost', 0)
        test_c = np.array(c)
        test_c = [test_c[np.newaxis] for i in range(batch_size)]
        test_c= np.concatenate(test_c, axis=0)
        self.cum_cost += test_c.sum()

        #save and log
        self.pool.store_multiple(test_obs, a, test_nextobs, test_rew, v_t, test_c, vc_t, logp_t, pi_info_t, test_term)
        self.logger.store(VVals=v_t, CostVVals=vc_t)
        
        # # debug !
        # self.pool_debug.store(self._current_observation, a[0], next_observation, reward, v_t[0], c, vc_t[0], logp_t[0], {k:v[0] for k,v in pi_info_t.items()}, terminal)
        # if terminal:
        #     last_val = self.policy.get_v(self._current_observation)
        #     last_cval = self.policy.get_vc(self._current_observation)
        #     self.pool_debug.finish_path(last_val, last_cval)


        self._path_length[self._alive_paths] += 1
        self._path_return[self._alive_paths] += reward
        self._path_cost[self._alive_paths] += c
        self._total_samples += self._alive_paths.sum()

        #### add to pool only after full epoch or terminal path
        ## working with masks for ending trajectories
        path_end_mask = self._path_length >= self._max_path_length
        if test_term.any() or path_end_mask.any():

            # init final values and quantify according to termination type
            last_val, last_cval = np.zeros(self.batch_size), np.zeros(self.batch_size)

            # If trajectory didn't reach terminal state, bootstrap value target(s)
            prem_term_mask = np.logical_not(path_end_mask)*test_term
            mat_term_mask = np.logical_or(test_term, path_end_mask) * np.logical_not(prem_term_mask)

            # Note: we do not count env time out as true terminal state
            last_val[prem_term_mask] = np.zeros(last_val[prem_term_mask].shape)
            last_cval[prem_term_mask] = np.zeros(last_cval[prem_term_mask].shape)
            
            # mature termination upon path end or env episode end
            if mat_term_mask.any():
                if self.policy.agent.reward_penalized:
                    last_val[mat_term_mask] = np.squeeze(self.policy.get_v(test_obs[mat_term_mask]))
                    last_cval[mat_term_mask] = np.zeros(last_cval[mat_term_mask].shape)
                else:
                    last_val[mat_term_mask] = np.squeeze(self.policy.get_v(test_obs[mat_term_mask]))
                    last_cval[mat_term_mask] = np.squeeze(self.policy.get_vc(test_obs[mat_term_mask]))

            
            term_mask = np.logical_or(prem_term_mask, mat_term_mask)
            non_term_mask = np.logical_not(term_mask)
            self._alive_paths = self._alive_paths*non_term_mask
            
            self.pool.finish_path_multiple(term_mask, last_val, last_cval)
            

            # Only save EpRet / EpLen if trajectory finished
            if not self._alive_paths.any():
                #@anyboby TODO handle logger /diagnostics for model
                # self.logger.store(RetEp=self._path_return, EpLen=self._path_length, CostEp=self._path_cost)
                print('debug')
            else:
                print('Warning: trajectory cut off by epoch at %d steps.'%self._path_length[term_mask][0])

            self._max_path_return = max(self._max_path_return,
                                        max(self._path_return))
            self.policy.reset() #does nohing for cpo policy atm
            self._current_observation = next_observation

            # reset if all paths have died
            if not self._alive_paths.any():
                self._current_observation = None
                self._path_length = np.zeros(self.batch_size)
                self._path_return = np.zeros(batch_size)
                self._path_cost = np.zeros(batch_size)
                self._alive_paths = np.ones(batch_size, dtype=np.bool)
                self._n_episodes = 0
            else:
                self._current_observation = next_observation

        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

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
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CostCumulative', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)

        # Value function values
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('CostVVals', with_min_and_max=True)

        # Pi loss and change
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossPiDelta', average_only=True)

        # Surr cost and change
        logger.log_tabular('SurrCost', average_only=True)
        logger.log_tabular('SurrCostDelta', average_only=True)

        # V loss and change
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossVDelta', average_only=True)

        # Time and steps elapsed
        logger.log_tabular('TotalEnvInteracts', self._total_samples)
        #logger.log_tabular('Time', time.time()-start_time)
        
