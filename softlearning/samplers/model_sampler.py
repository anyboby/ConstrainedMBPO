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
EPS = 1e-8

class ModelSampler(CpoSampler):
    def __init__(self,
                 max_path_length,
                 batch_size=1000,
                 store_last_n_paths = 10,
                 preprocess_type='default',
                 max_uncertainty = 1e8,
                 logger = None):
        self._max_path_length = max_path_length
        self._path_length = np.zeros(batch_size)
        self._path_return = np.zeros(batch_size)
        self._path_cost = np.zeros(batch_size)
        self._path_uncertainty = np.zeros(batch_size)

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
        self._cum_cost = 0
        self._max_uncertainty = max_uncertainty
        self._total_Vs = 0
        self._total_CVs = 0
        self._cum_var = 0
        self._cum_var_v = 0
        self._cum_var_vc = 0



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
        mean_rollout_length = self._total_samples / (self.batch_size+EPS)
        mean_ensemble_dkl_cum = np.mean(self._path_uncertainty)
        mean_ensemble_dkl = np.mean((self._path_uncertainty.sum()+1e-8)/(self._path_length.sum()+EPS))
        mean_ensemble_var = self._cum_var/(self._total_samples+EPS)
        cost_rate = self._cum_cost/(self._total_samples+EPS)
        return_rate = self._path_return.sum()/(self._total_samples+EPS)
        VVals_mean = self._total_Vs / (self._total_samples+EPS)
        VVals_var = self._cum_var_v / (self._total_samples+EPS)

        CostV_mean = self._total_CVs / (self._total_samples+EPS)
        CostV_var = self._cum_var_vc / (self._total_samples+EPS)

        diagnostics.update({
            'samples_added': self._total_samples,
            'rollout_length_max': self._n_episodes,
            'rollout_length_mean': mean_rollout_length,
            'ensemble_dkl_mean': mean_ensemble_dkl,
            'ensemble_dkl_cum_mean' : mean_ensemble_dkl_cum,
            'ensemble_var_mean' : mean_ensemble_var,
            'cost_rate': cost_rate,
            'return_rate': return_rate,
            'VVals':VVals_mean,
            'VVals_var':VVals_var,
            'CostVVals':CostV_mean,
            'CostV_var':CostV_var,
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
        self.policy.reset() #does nohing for cpo policy atm
        self._path_length = np.zeros(self.batch_size)
        self._path_return = np.zeros(self.batch_size)
        self._path_cost = np.zeros(self.batch_size)
        self._path_uncertainty = np.zeros(self.batch_size)
        self._total_samples = 0
        self._n_episodes = 0
        self._total_Vs = 0
        self._total_CVs = 0
        self._cum_cost = 0
        self._cum_var = 0
        self._cum_var_v = 0
        self._cum_var_vc = 0

    def sample(self):
        assert self.pool.has_room           #pool full! empty before sampling.
        assert self._current_observation is not None # reset before sampling !
        assert self.pool.alive_paths.any()  # reset before sampling !

        self._n_episodes += 1
        alive_paths = self.pool.alive_paths
        current_obs = self._current_observation

        # Get outputs from policy
        get_action_outs = self.policy.get_action_outs(current_obs)

        a = get_action_outs['pi']
        v_t = get_action_outs['v']
        vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']

        ##### @anyboby temporary
        ### unpack ensemble outputs
        v_var = np.var(v_t, axis=0)
        vc_var = np.var(vc_t, axis=0)

        v_t = np.mean(v_t, axis=0)
        vc_t = np.mean(vc_t, axis=0)
        #####

        next_obs, reward, terminal, info = self.env.step(current_obs, a)
        reward = np.squeeze(reward, axis=-1)
        terminal = np.squeeze(terminal, axis=-1)
        c = info.get('cost', np.zeros(reward.shape))
        en_disag = info.get('ensemble_disagreement', 0)
        self._cum_var += info.get('ensemble_var', 0)*len(self.pool.alive_paths)

        ## ____________________________________________ ##
        ##    Check Uncertainty f. each Trajectory      ##
        ## ____________________________________________ ##

        ### check if too uncertain before storing info of the taken step
        path_uncertainty = self._path_uncertainty[alive_paths] + en_disag
        
        ### too_uncertain_mask = path_uncertainty > self._max_uncertainty
        ### testing
        too_uncertain_mask = path_uncertainty > vc_var*3
        
        
        ### finish too uncertain paths before storing info of the taken step
        # remaining_paths refers to the paths we have finished and has the same shape 
        # as our terminal mask (too_uncertain_mask)
        # alive_paths refers to all original paths and therefore has shape batch_size
        remaining_paths = self._finish_paths(too_uncertain_mask, append_vals=True)
        alive_paths = self.pool.alive_paths
        
        ## ____________________________________________ ##
        ##    Store Info of the remaining paths         ##
        ## ____________________________________________ ##

        
        current_obs = current_obs[remaining_paths]
        a = a[remaining_paths]
        next_obs = next_obs[remaining_paths]
        reward = reward[remaining_paths]
        v_t = v_t[remaining_paths]
        c = c[remaining_paths]
        vc_t = vc_t[remaining_paths]
        logp_t = logp_t[remaining_paths]
        pi_info_t = {k:v[remaining_paths] for k,v in pi_info_t.items()}
        terminal = terminal[remaining_paths]
        en_disag = en_disag[remaining_paths]

        ### Store info in pool
        self.pool.store_multiple(current_obs,
                                a,
                                next_obs,
                                reward,
                                v_t,
                                c,
                                vc_t,
                                logp_t,
                                pi_info_t,
                                terminal)

        #self.logger.store(VVals=v_t, CostVVals=vc_t, VVars=v_var)
        self._total_Vs += v_t.sum()
        self._total_CVs += vc_t.sum()

        # # debug !
        # self.pool_debug.store(self._current_observation, a[0], next_observation, reward, v_t[0], c, vc_t[0], logp_t[0], {k:v[0] for k,v in pi_info_t.items()}, terminal)
        # if terminal:
        #     last_val = self.policy.get_v(self._current_observation)
        #     last_cval = self.policy.get_vc(self._current_observation)
        #     self.pool_debug.finish_path(last_val, last_cval)

        #### update some sampler infos
        self._path_uncertainty[alive_paths] += en_disag
        self._path_length[alive_paths] += 1
        self._path_return[alive_paths] += reward
        self._path_cost[alive_paths] += c
        self._total_samples += alive_paths.sum()
        self._cum_cost += c.sum()
        self._cum_var += info.get('ensemble_var', 0)*alive_paths.sum()
        self._cum_var_v += v_var.sum()
        self._cum_var_vc += vc_var.sum()
        self._max_path_return = max(self._max_path_return,
                            max(self._path_return))

        #### terminate mature termination due to path length
        ## update obs before finishing paths (_finish_paths() uses current obs)
        self._current_observation = next_obs
        path_end_mask = (self._path_length >= self._max_path_length-1)[alive_paths]
        remaining_paths = self._finish_paths(term_mask=path_end_mask, append_vals=True)
        
        ## update remaining paths and obs
        self._current_observation = self._current_observation[remaining_paths]

        #### terminate real termination due to env end
        prem_term_mask = terminal[remaining_paths]
        remaining_paths = self._finish_paths(term_mask=prem_term_mask, append_vals=False)

        ### update alive paths
        alive_paths = self.pool.alive_paths
        self._current_observation = self._current_observation[remaining_paths]

        alive_ratio = sum(alive_paths)/self.batch_size
        info['alive_ratio'] = alive_ratio

        return next_obs, reward, terminal, info


    def _finish_paths(self, term_mask, append_vals=False):
        """
        terminates paths that are indicated in term_mask. Append_vals should be set to 
        True/False to indicate, whether values of the current states of those paths should 
        be appended (Note: Premature termination due to environment term should not 
        include appended values, while Mature termination upon path length excertion should 
        include appended values)

        Warning! throws error if trying to terminate an already terminated path. 

        Args:
            term_mask: Mask with the shape of the currently alive paths that indicates which 
                paths should be termianted
            append_vals: True/False whether values of the current state should be appended
        
        Returns: 
            remaining_mask: A Mask that indicates the remaining alive paths. Has the same shape 
                as the arg term_mask
        """
        if not term_mask.any():
            return np.logical_not(term_mask)

        # init final values
        last_val, last_cval = np.zeros(shape=term_mask.shape), np.zeros(shape=term_mask.shape)

        ## rebase last_val and last_cval to terminating paths
        last_val = last_val[term_mask]
        last_cval = last_cval[term_mask]

        cur_obs = self._current_observation

        # We do not count env time out (mature termination) as true terminal state, append values
        if append_vals:
            if self.policy.agent.reward_penalized:
                last_val = np.squeeze(np.mean(self.policy.get_v(cur_obs[term_mask]), axis=0))
                #last_val = np.squeeze(self.policy.get_v(cur_obs[term_mask]))

            else:
                last_val = np.squeeze(np.mean(self.policy.get_v(cur_obs[term_mask]), axis=0))
                last_cval = np.squeeze(np.mean(self.policy.get_vc(cur_obs[term_mask]), axis=0))
                # last_val = np.squeeze(self.policy.get_v(cur_obs[term_mask]))
                # last_cval = np.squeeze(self.policy.get_vc(cur_obs[term_mask]))

        self.pool.finish_path_multiple(term_mask, last_val, last_cval)
        remaining_path_mask = np.logical_not(term_mask)

        return remaining_path_mask
        
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
                last_val = np.squeeze(np.mean(self.policy.get_v(current_obs), axis=0))
                # last_val = np.squeeze(self.policy.get_v(current_obs))
            else:
                last_val = np.squeeze(np.mean(self.policy.get_v(current_obs), axis=0))
                last_cval = np.squeeze(np.mean(self.policy.get_vc(current_obs), axis=0))
                # last_val = np.squeeze(self.policy.get_v(current_obs))
                # last_cval = np.squeeze(self.policy.get_vc(current_obs))

            self.pool.finish_path_multiple(term_mask, last_val, last_cval)

        alive_paths = self.pool.alive_paths
        assert alive_paths.sum()==0   ## something went wrong with finishing all paths

        if not alive_paths.any():
            #@anyboby TODO handle logger /diagnostics for model
            # self.logger.store(RetEp=self._path_return, EpLen=self._path_length, CostEp=self._path_cost)
            print('All trajectories finished. @anyboby implement some logging magic here')
        
        return self.get_diagnostics()

    def log(self):
        """
        logs several stats over the timesteps since the last 
        flush (such as epCost, totalCost etc.)
        """
        logger = self.logger
        cumulative_cost = mpi_sum(self._cum_cost)    
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
        #