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
                 batch_size=1000,
                 store_last_n_paths = 10,
                 preprocess_type='default',
                 logger = None):
        self._max_path_length = max_path_length
        self._path_length = np.zeros(batch_size)
        self._path_return = np.zeros(batch_size)
        self._path_cost = np.zeros(batch_size)
        self._path_uncertainty = np.zeros(batch_size)


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
        self._max_uncertainty = 0.005

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

    def reset(self, observations):
        self._current_observation = observations
        self._path_length = np.zeros(self.batch_size)
        self._path_return = np.zeros(self.batch_size)
        self._path_cost = np.zeros(self.batch_size)
        self._path_uncertainty = np.zeros(self.batch_size)
        self._n_episodes = 0


    def sample(self):
        assert self.pool.has_room           #pool full! empty before sampling.
        assert self._current_observation is not None # reset before sampling !

        self._n_episodes += 1
        batch_size = self.batch_size
        alive_paths = self.pool.alive_paths
        current_obs = self._current_observation

        # Get outputs from policy
        get_action_outs = self.policy.get_action_outs(current_obs)

        a = get_action_outs['pi']
        v_t = get_action_outs['v']
        vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']

        next_obs, reward, terminal, info = self.env.step(current_obs, a)
        reward = np.squeeze(reward)
        terminal = np.squeeze(terminal)
        
        c = info.get('cost', np.zeros(reward.shape))
        self.cum_cost += c.sum()


        #save and log
        self.pool.store_multiple(current_obs, a, next_obs, reward, v_t, c, vc_t, logp_t, pi_info_t, terminal)
        self.logger.store(VVals=v_t, CostVVals=vc_t)
        
        # # debug !
        # self.pool_debug.store(self._current_observation, a[0], next_observation, reward, v_t[0], c, vc_t[0], logp_t[0], {k:v[0] for k,v in pi_info_t.items()}, terminal)
        # if terminal:
        #     last_val = self.policy.get_v(self._current_observation)
        #     last_cval = self.policy.get_vc(self._current_observation)
        #     self.pool_debug.finish_path(last_val, last_cval)

        en_disag = info.get('ensemble_disagreement', 0)

        self._path_uncertainty[alive_paths] += en_disag
        self._path_length[alive_paths] += 1
        self._path_return[alive_paths] += reward
        self._path_cost[alive_paths] += c
        self._total_samples += alive_paths.sum()

        #### add to pool only after full epoch or terminal path
        ## working with masks for ending trajectories
        path_end_mask = (self._path_length >= self._max_path_length)[alive_paths]
        too_uncertain_mask = self._path_uncertainty[alive_paths] > self._max_uncertainty
        path_end_mask = np.logical_or(path_end_mask, too_uncertain_mask)
        if terminal.any() or path_end_mask.any():


            # If trajectory didn't reach terminal state, bootstrap value target(s)
            prem_term_mask = np.logical_not(path_end_mask)*terminal
            mat_term_mask = np.logical_or(terminal, path_end_mask) * np.logical_not(prem_term_mask)
            term_mask = np.logical_or(prem_term_mask, mat_term_mask)
            non_term_mask = np.logical_not(term_mask)

            # init final values and quantify according to termination type
            last_val, last_cval = np.zeros(shape=term_mask.shape), np.zeros(shape=term_mask.shape)

            # We do not count env time out (mature termination) as true terminal state, append values
            if mat_term_mask.any():
                if self.policy.agent.reward_penalized:
                    last_val[mat_term_mask] = np.squeeze(self.policy.get_v(next_obs[mat_term_mask]))
                else:
                    last_val[mat_term_mask] = np.squeeze(self.policy.get_v(next_obs[mat_term_mask]))
                    last_cval[mat_term_mask] = np.squeeze(self.policy.get_vc(next_obs[mat_term_mask]))

            ## rebase last_val and last_cval to paths that are terminated
            last_val = last_val[term_mask]
            last_cval = last_cval[term_mask]

            self.pool.finish_path_multiple(term_mask, last_val, last_cval)

            alive_paths = self.pool.alive_paths

            # Only save EpRet / EpLen if trajectory finished
            if not alive_paths.any():
                #@anyboby TODO handle logger /diagnostics for model
                # self.logger.store(RetEp=self._path_return, EpLen=self._path_length, CostEp=self._path_cost)
                print('All trajectories dead. @anyboby implement some logging magic here')
            else:
                print('Warning: trajectory cut off by epoch at %d steps.'%self._path_length[alive_paths][0])

            self._max_path_return = max(self._max_path_return,
                                        max(self._path_return))
            self.policy.reset() #does nohing for cpo policy atm

            # reset if all paths have died
            if not alive_paths.any():
                self._current_observation = None
                self._path_length = np.zeros(batch_size)
                self._path_return = np.zeros(batch_size)
                self._path_uncertainty = np.zeros(batch_size)
                self._path_cost = np.zeros(batch_size)
                self._alive_paths = np.ones(batch_size, dtype=np.bool)
                self._n_episodes = 0
            else:
                self._current_observation = next_obs[non_term_mask]

        else:
            self._current_observation = next_obs

        info['alive_ratio'] = alive_paths.sum()/self.batch_size

        return next_obs, reward, terminal, info

    def finish_all_paths(self):

        alive_paths=self.pool.alive_paths ##any paths that are still alive did not terminate by env
        # init final values and quantify according to termination type
        # Note: we do not count env time out as true terminal state
        if self._current_observation is None: return
        assert len(self._current_observation) == alive_paths.sum()
        current_obs = self._current_observation

        if alive_paths.any():
            last_val, last_cval = np.zeros(shape=alive_paths.sum()), np.zeros(shape=alive_paths.sum())
            term_mask = np.ones(shape=alive_paths.sum(), dtype=np.bool)
            if self.policy.agent.reward_penalized:
                last_val = np.squeeze(self.policy.get_v(current_obs))
            else:
                last_val = np.squeeze(self.policy.get_v(current_obs))
                last_cval = np.squeeze(self.policy.get_vc(current_obs))

            self.pool.finish_path_multiple(term_mask, last_val, last_cval)

        alive_paths = self.pool.alive_paths
        assert alive_paths.sum()==0   ## check if all paths have been finished

        # Only save EpRet / EpLen if trajectory finished
        if not alive_paths.any():
            #@anyboby TODO handle logger /diagnostics for model
            # self.logger.store(RetEp=self._path_return, EpLen=self._path_length, CostEp=self._path_cost)
            print('All trajectories finished. @anyboby implement some logging magic here')

        self._max_path_return = max(self._max_path_return,
                                    max(self._path_return))
        self.policy.reset() #does nohing for cpo policy atm

        self._current_observation = None
        self._path_length = np.zeros(self.batch_size)
        self._path_return = np.zeros(self.batch_size)
        self._path_uncertainty = np.zeros(self.batch_size)
        self._path_cost = np.zeros(self.batch_size)
        self._alive_paths = np.ones(self.batch_size, dtype=np.bool)
        self._n_episodes = 0


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
        
