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
                 max_uncertainty_c = 1e3,
                 max_uncertainty_r = 1e-2,
                 use_inv_var = False,
                 logger = None):
        self._max_path_length = max_path_length
        self._path_length = np.zeros(batch_size)
        self._path_return = np.zeros(batch_size)
        self._path_cost = np.zeros(batch_size)
        self._path_return_var = np.zeros(batch_size)
        self._path_cost_var = np.zeros(batch_size)
        self._path_dyn_var = np.zeros(batch_size)

        self.use_inv_var = use_inv_var

        if logger:
            self.logger = logger
        else: 
            self.logger = EpochLogger()

        self._store_last_n_paths = store_last_n_paths
        self._last_n_paths = deque(maxlen=store_last_n_paths)

        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._current_observation = None
        self._last_action = None
        self._max_uncertainty_c = max_uncertainty_c
        self._max_uncertainty_rew = max_uncertainty_r,

        self._total_samples = 0
        self._n_episodes = 0
        self._total_Vs = 0
        self._total_CVs = 0
        self._total_rew = 0
        self._total_rew_var = 0
        self._total_cost = 0
        self._total_cost_var = 0
        self._total_dyn_var = 0
        self._total_V_var = 0
        self._total_CV_var = 0
        self._total_dkl_mean_dyn = 0
        self._total_dkl_med_dyn = 0

        self.batch_size = batch_size

        self._obs_process_type = preprocess_type
        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool
        self.ensemble_size = env.num_networks

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

        ensemble_rew_var_perstep = self._total_rew_var/(self._total_samples+EPS)
        ensemble_cost_var_perstep = self._total_cost_var/(self._total_samples+EPS)
        ensemble_dyn_var_perstep = self._total_dyn_var/(self._total_samples+EPS)
        
        VVals_mean = self._total_Vs / (self._total_samples+EPS)
        VVals_var = self._total_V_var / (self._total_samples+EPS)

        CostV_mean = self._total_CVs / (self._total_samples+EPS)
        CostV_var = self._total_CV_var / (self._total_samples+EPS)

        Dyn_Dkl_mean = self._total_dkl_mean_dyn / (self._total_samples+EPS)
        Dyn_Dkl_med = self._total_dkl_med_dyn / (self._total_samples+EPS)

        diagnostics.update({
            'samples_added': self._total_samples,
            'rollout_length_max': self._n_episodes,
            'rollout_length_mean': mean_rollout_length,
            'ensemble_rew_var_perstep': ensemble_rew_var_perstep,
            'ensemble_cost_var_perstep' : ensemble_cost_var_perstep,
            'ensemble_dyn_var_perstep' : ensemble_dyn_var_perstep,
            'ensemble_VVals':VVals_mean,
            'ensemble_VVals_var':VVals_var,
            'ensemble_CostVVals':CostV_mean,
            'ensemble_CostV_var':CostV_var,
            'ensemble_dyn_DKL_mean': Dyn_Dkl_mean,
            'ensemble_dyn_DKL_med': Dyn_Dkl_med,
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
        self._current_observation = np.tile(observations[None], (self.ensemble_size, 1, 1))

        self.policy.reset() #does nohing for cpo policy atm
        self._path_length = np.zeros(self.batch_size)
        self._path_return = np.zeros(shape=(self.ensemble_size, self.batch_size))
        self._path_cost = np.zeros(shape=(self.ensemble_size, self.batch_size))
        self._path_return_var = np.zeros(self.batch_size)
        self._path_cost_var = np.zeros(self.batch_size)
        self._path_dyn_var = np.zeros(self.batch_size)
        
        self.model_inds = 0 # self.env.random_inds(size=1)[0]

        self._total_samples = 0
        self._n_episodes = 0
        self._total_Vs = 0
        self._total_CVs = 0
        self._total_cost = 0
        self._total_cost_var = 0
        self._total_rew = 0
        self._total_rew_var = 0
        self._total_dyn_var = 0
        self._total_V_var = 0
        self._total_CV_var = 0
        self._total_dkl_mean_dyn = 0
        self._total_dkl_med_dyn = 0

    def sample(self):
        assert self.pool.has_room           #pool full! empty before sampling.
        assert self._current_observation is not None # reset before sampling !
        assert self.pool.alive_paths.any()  # reset before sampling !

        self._n_episodes += 1
        alive_paths = self.pool.alive_paths
        current_obs = self._current_observation

        # Get outputs from policy
        get_action_outs = self.policy.get_action_outs(current_obs, factored=True, inc_var=True)
        
        a = get_action_outs['pi']
        v_t = get_action_outs['v']
        vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']

        ##### @anyboby temporary
        ### unpack ensemble outputs, if gaussian
        v_var = get_action_outs.get('v_var', np.tile(np.var(v_t, axis=0)[None], reps=(self.ensemble_size, 1)))
        vc_var = get_action_outs.get('vc_var', np.tile(np.var(vc_t, axis=0)[None], reps=(self.ensemble_size, 1))) 
        #####

        ## ____________________________________________ ##
        ##                      Step                    ##
        ## ____________________________________________ ##

        next_obs, reward, terminal, info = self.env.step(current_obs, a)

        reward = np.squeeze(reward, axis=-1)
        
        rew_var = info.get('rew_ensemble_var', np.zeros(reward.shape))

        c = info.get('cost', np.zeros(reward.shape))
        c_var = info.get('cost_ensemble_var', np.zeros(reward.shape))

        terminal = np.squeeze(terminal, axis=-1)

        dkl_mean_dyn = info.get('dyn_ensemble_dkl_mean', 0)
        dkl_med_dyn = info.get('dyn_ensemble_dkl_med', 0)
        dyn_var = info.get('dyn_ensemble_var', np.zeros(reward.shape))

        next_val, next_val_var = self.policy.get_v(next_obs, factored=True, inc_var=True)

        next_cval, next_cval_var = self.policy.get_vc(next_obs, factored=True, inc_var=True)

        #### variance for gaussian mixture, add dispersion of means to variance
        next_val_var = np.mean(next_val_var, axis=0) + np.mean(next_val**2, axis=0) - (np.mean(next_val, axis=0))**2
        next_cval_var = np.mean(next_cval_var, axis=0) + np.mean(next_cval**2, axis=0) - (np.mean(next_cval, axis=0))**2
        c_var = np.mean(c, axis=0) + np.mean(c**2, axis=0) - (np.mean(c_var, axis=0))**2

        ## ____________________________________________ ##
        ##    Check Uncertainty f. each Trajectory      ##
        ## ____________________________________________ ##


        ### check if too uncertain before storing info of the taken step 
        ### (so we don't take a "bad step" by appending values of next state)
        cost_uncertainty = self._path_cost_var[alive_paths] + c_var
        rew_uncertainty = self._path_return_var[alive_paths] + rew_var 

        ### running means of variances
        cost_var_rm = self._total_cost_var+EPS**2/(self._total_samples+EPS)
        rew_var_rm = self._total_rew_var+EPS**2/(self._total_samples+EPS)

        # too_uncertain_mask = path_uncertainty > self._max_uncertainty
        too_uncertain_paths = np.logical_or(cost_uncertainty + next_cval_var > self._max_uncertainty_c, \
                                            rew_uncertainty + next_val_var > self._max_uncertainty_rew) 
        # too_uncertain_mask_path = np.logical_or(cost_uncertainty > next_cval_var, \
        #                                     rew_uncertainty > next_val_var) 

        ### finish too uncertain paths before storing info of the taken step
        # remaining_paths refers to the paths we have finished and has the same shape 
        # as our terminal mask (too_uncertain_mask)
        # alive_paths refers to all original paths and therefore has shape batch_size
        remaining_paths = self._finish_paths(too_uncertain_paths, append_vals=True)
        alive_paths = self.pool.alive_paths
        if not alive_paths.any():
            info['alive_ratio'] = 0
            return next_obs, reward, terminal, info

        ## ____________________________________________ ##
        ##    Store Info of the remaining paths         ##
        ## ____________________________________________ ##

        current_obs     = current_obs[:,remaining_paths]
        a               = a[:,remaining_paths]
        next_obs        = next_obs[:,remaining_paths]
        reward          = reward[:,remaining_paths]
        rew_var         = rew_var[remaining_paths]
        v_t             = v_t[:,remaining_paths]
        v_var           = v_var[:,remaining_paths]

        cost_uncertainty = cost_uncertainty[remaining_paths]
        rew_uncertainty = rew_uncertainty[remaining_paths]
        c               = c[:,remaining_paths]
        c_var           = c_var[remaining_paths]
        vc_t            = vc_t[:, remaining_paths]
        vc_var          = vc_var[:, remaining_paths]
        dyn_var         = dyn_var[remaining_paths]
        
        terminal        = terminal[:,remaining_paths]

        logp_t          = logp_t[:,remaining_paths]
        pi_info_t       = {k:v[:,remaining_paths] for k,v in pi_info_t.items()}
        # pi_info_t       = {k:v[self.model_inds] for k,v in pi_info_t.items()}


        #### update some sampler infos
        self._total_samples += alive_paths.sum()

        self._total_cost += c[self.model_inds].sum()
        self._total_rew += reward[self.model_inds].sum()

        self._path_length[alive_paths] += 1
        self._path_return[:, alive_paths] += reward
        self._path_cost[:, alive_paths] += c
        self._path_return_var[alive_paths] = np.var(self._path_return[:, alive_paths], axis=0)
        self._path_cost_var[alive_paths] = np.var(self._path_cost[:, alive_paths], axis=0)

        self._path_dyn_var[alive_paths] += dyn_var
        self._total_cost_var += cost_uncertainty.sum()
        self._total_dyn_var += dyn_var.sum()
        self._total_rew_var += rew_uncertainty.sum()

        self._total_Vs += v_t[self.model_inds].sum()
        self._total_CVs += vc_t[self.model_inds].sum()
        self._total_V_var += v_var[self.model_inds].sum()
        self._total_CV_var += vc_var[self.model_inds].sum()
        
        self._total_dkl_mean_dyn += dkl_mean_dyn*alive_paths.sum()
        self._total_dkl_med_dyn += dkl_med_dyn*alive_paths.sum()

        self._max_path_return = max(self._max_path_return,
                            np.max(self._path_return))

        #### only store one trajectory in buffer 
        self.pool.store_multiple(current_obs,
                                        a,
                                        next_obs,
                                        reward,
                                        v_t,
                                        v_var,
                                        c,
                                        vc_t,
                                        vc_var,
                                        logp_t,
                                        pi_info_t,
                                        terminal)

        #### terminate mature termination due to path length
        ## update obs before finishing paths (_finish_paths() uses current obs)
        self._current_observation = next_obs

        path_end_mask = (self._path_length >= self._max_path_length-1)[alive_paths]
        remaining_paths = self._finish_paths(term_mask=path_end_mask, append_vals=True)
        
        ## update remaining paths and obs
        self._current_observation = self._current_observation[:,remaining_paths]

        #### terminate real termination due to env end
        prem_term_mask = np.any(terminal[:,remaining_paths], axis=0)            ##@anyboby maybe check later, if terminal per model should be possible
        remaining_paths = self._finish_paths(term_mask=prem_term_mask, append_vals=False)

        ### update alive paths
        alive_paths = self.pool.alive_paths
        self._current_observation = self._current_observation[:,remaining_paths]

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

        # cur_obs = self._current_observation[self.model_inds]

        # We do not count env time out (mature termination) as true terminal state, append values
        if append_vals:
            if self.policy.agent.reward_penalized:
                last_val, last_val_var = self.policy.get_v(self._current_observation, factored=True, inc_var=True)
            else:
                last_val, last_val_var = self.policy.get_v(self._current_observation, factored=True, inc_var=True)
                last_cval, last_cval_var = self.policy.get_vc(self._current_observation, factored=True, inc_var=True)
        else:
            # init final values
            last_val, last_cval = np.zeros(shape=term_mask.shape), np.zeros(shape=term_mask.shape)
            last_val_var, last_cval_var = np.zeros(shape=term_mask.shape), np.zeros(shape=term_mask.shape)

            ## rebase last_val and last_cval to terminating paths
            last_val, last_cval = last_val[term_mask], last_cval[term_mask]
            last_val_var, last_cval_var = last_val_var[term_mask], last_cval_var[term_mask]

        # last_val_var = np.mean(last_val_var, axis=0) + np.mean(last_val**2, axis=0) - (np.mean(last_val, axis=0))**2
        # last_cval_var = np.mean(last_cval_var, axis=0) + np.mean(last_cval**2, axis=0) - (np.mean(last_cval, axis=0))**2

        self.pool.finish_path_multiple(term_mask, last_val, last_val_var, last_cval, last_cval_var)

        remaining_path_mask = np.logical_not(term_mask)

        return remaining_path_mask
        
    def finish_all_paths(self):

        alive_paths=self.pool.alive_paths ##any paths that are still alive did not terminate by env
        # init final values and quantify according to termination type
        # Note: we do not count env time out as true terminal state
        if not alive_paths.any(): return self.get_diagnostics()
        assert self._current_observation.shape[1] == alive_paths.sum()
        #assert self.model_ind_mask.shape[1] == alive_paths.sum()

        if alive_paths.any():
            term_mask = np.ones(shape=alive_paths.sum(), dtype=np.bool)
            # cur_obs = self._current_observation[self.model_inds]

            if self.policy.agent.reward_penalized:
                last_val, last_val_var = self.policy.get_v(self._current_observation, factored=True, inc_var=True)
            else:
                last_val, last_val_var = self.policy.get_v(self._current_observation, factored=True, inc_var=True)
                last_cval, last_cval_var = self.policy.get_vc(self._current_observation, factored=True, inc_var=True)

            # last_val_var = np.mean(last_val_var, axis=0) + np.mean(last_val**2, axis=0) - (np.mean(last_val, axis=0))**2
            # last_cval_var = np.mean(last_cval_var, axis=0) + np.mean(last_cval**2, axis=0) - (np.mean(last_cval, axis=0))**2

            self.pool.finish_path_multiple(term_mask, last_val, last_val_var, last_cval, last_cval_var)
            
        alive_paths = self.pool.alive_paths
        assert alive_paths.sum()==0   ## something went wrong with finishing all paths
        
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