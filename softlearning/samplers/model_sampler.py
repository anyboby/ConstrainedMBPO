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
                 cares_about_cost = False,
                 max_uncertainty_c = 3,
                 max_uncertainty_r = 3,
                 rollout_mode = False,
                 logger = None):
        self._max_path_length = max_path_length
        self._path_length = np.zeros(batch_size)
        self._path_return = np.zeros(batch_size)
        self._path_cost = np.zeros(batch_size)
        self._path_return_var = np.zeros(batch_size)
        self._path_cost_var = np.zeros(batch_size)
        self._path_dyn_var = np.zeros(batch_size)

        self.cares_about_cost = cares_about_cost
        self.rollout_mode = rollout_mode

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
        self._path_dyn_var = 0
        self._total_dkl_med_dyn = 0

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

        ensemble_cost_rate = np.sum(np.mean(self._path_cost, axis=0))/(self._total_samples+EPS)
        ensemble_rew_rate = np.sum(np.mean(self._path_return, axis=0))/(self._total_samples+EPS)

        vals_mean = self._total_Vs / (self._total_samples+EPS)

        cval_mean = self._total_CVs / (self._total_samples+EPS)

        dyn_Dkl_med = self._total_dkl_med_dyn / (self._total_samples+EPS)

        diagnostics.update({
            'samples_added': self._total_samples,
            'rollout_length_max': self._n_episodes,
            'rollout_length_mean': mean_rollout_length,
            'ensemble_ep_rew_var_perstep': ensemble_rew_var_perstep,
            'ensemble_ep_cost_var_perstep' : ensemble_cost_var_perstep,
            'ensemble_ep_dyn_var_perstep' : ensemble_dyn_var_perstep,
            'ensemble_cost_rate' : ensemble_cost_rate,
            'ensemble_rew_rate' : ensemble_rew_rate,
            'ensemble_v_mean':vals_mean,
            'ensemble_cv_mean':cval_mean,
            'ensemble_dyn_DKL_med': dyn_Dkl_med,
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
        self.batch_size = observations.shape[0]

        self._starting_uncertainty = np.zeros(self.batch_size)

        if self.rollout_mode == 'iv_gae':
            self._current_observation = np.tile(observations[None], (self.ensemble_size, 1, 1))
        else:
            self._current_observation = observations

        self.policy.reset() #does nohing for cpo policy atm
        self.pool.reset(self.batch_size, self.env.dyn_target_var)

        self._path_length = np.zeros(self.batch_size)
        if self.rollout_mode=='iv_gae':
            self._path_return = np.zeros(shape=(self.ensemble_size, self.batch_size))
            self._path_cost = np.zeros(shape=(self.ensemble_size, self.batch_size))
        else:
            self._path_return = np.zeros(shape=(self.batch_size))
            self._path_cost = np.zeros(shape=(self.batch_size))

        self._path_return_var = np.zeros(self.batch_size)
        self._path_cost_var = np.zeros(self.batch_size)
        self._path_dyn_var = np.zeros(self.batch_size)
        
        self.model_inds = np.random.randint(self.ensemble_size)

        self._total_samples = 0
        self._n_episodes = 0
        self._total_Vs = 0
        self._total_CVs = 0
        self._total_cost = 0
        self._total_cost_var = 0
        self._total_rew = 0
        self._total_rew_var = 0
        self._total_dyn_var = 0
        self._total_dkl_med_dyn = 0

    def sample(self, max_samples=None):
        assert self.pool.has_room           #pool full! empty before sampling.
        assert self._current_observation is not None # reset before sampling !
        assert self.pool.alive_paths.any()  # reset before sampling !

        self._n_episodes += 1
        alive_paths = self.pool.alive_paths
        current_obs = self._current_observation

        # Get outputs from policy
        get_action_outs = self.policy.get_action_outs(current_obs, factored=True, inc_var=True)
        
        a = get_action_outs['pi']
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']

        ##### @anyboby temporary
        ### unpack ensemble outputs, if gaussian
        if self.rollout_mode=='iv_gae':
            v_t = get_action_outs['v']
            vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func

            v_var = get_action_outs.get('v_var', 0)
            vc_var = get_action_outs.get('vc_var', 0) 
        else:
            v_t = np.mean(get_action_outs['v'], axis=0)
            vc_t = np.mean(get_action_outs.get('vc', 0), axis=0)  # Agent may not use cost value func

            v_var = np.mean(get_action_outs.get('v_var', 0), axis=0)
            vc_var = np.mean(get_action_outs.get('vc_var', 0), axis=0)
        #####

        ## ____________________________________________ ##
        ##                      Step                    ##
        ## ____________________________________________ ##

        next_obs, reward, terminal, info = self.env.step(current_obs, a)

        reward = np.squeeze(reward, axis=-1)
        c = np.squeeze(info.get('cost', np.zeros(reward.shape)))
        terminal = np.squeeze(terminal, axis=-1)
        dkl_med_dyn = info.get('dyn_ensemble_dkl_med', 0)
        dyn_ep_var = info.get('dyn_ep_var', np.zeros(shape=reward.shape[1:]))

        if self._n_episodes == 1:
            self._starting_uncertainty = np.mean(dyn_ep_var, axis=-1)
        ## ____________________________________________ ##
        ##    Check Uncertainty f. each Trajectory      ##
        ## ____________________________________________ ##


        ### check if too uncertain before storing info of the taken step 
        ### (so we don't take a "bad step" by appending values of next state)
        
        if self.rollout_mode=='uncertainty':
            too_uncertain_paths = np.mean(dyn_ep_var, axis=-1) > self._max_uncertainty_rew * self._starting_uncertainty[self.pool.alive_paths]
        else:
            too_uncertain_paths = np.zeros(shape=self.pool.alive_paths.sum(), dtype=np.bool)
        
        ### early terminate paths if max_samples is given
        if max_samples and not self.rollout_mode=='iv_gae':
            n = self._total_samples + alive_paths.sum() - too_uncertain_paths.sum()
            n = max(n-max_samples, 0)
            early_term = np.zeros_like(too_uncertain_paths[~too_uncertain_paths], dtype=np.bool)
            early_term[:n] = True
            too_uncertain_paths[~too_uncertain_paths] = early_term

        ### finish too uncertain paths before storing info of the taken step into buffer
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

        if self.rollout_mode=='iv_gae':
            current_obs     = current_obs[:,remaining_paths]
            a               = a[:,remaining_paths]
            next_obs        = next_obs[:,remaining_paths]
            reward          = reward[:,remaining_paths]
            v_t             = v_t[:,remaining_paths]
            v_var           = v_var[:,remaining_paths]

            c               = c[:,remaining_paths]
            vc_t            = vc_t[:, remaining_paths]
            vc_var          = vc_var[:, remaining_paths]
            terminal        = terminal[:,remaining_paths]

            logp_t          = logp_t[:,remaining_paths]
            pi_info_t       = {k:v[:,remaining_paths] for k,v in pi_info_t.items()}
        else:
            current_obs     = current_obs[remaining_paths]
            a               = a[remaining_paths]
            next_obs        = next_obs[remaining_paths]
            reward          = reward[remaining_paths]
            v_t             = v_t[remaining_paths]
            v_var           = v_var[remaining_paths]

            c               = c[remaining_paths]
            vc_t            = vc_t[remaining_paths]
            vc_var          = vc_var[remaining_paths]
            terminal        = terminal[remaining_paths]

            logp_t          = logp_t[remaining_paths]
            pi_info_t       = {k:v[remaining_paths] for k,v in pi_info_t.items()}

        dyn_ep_var      = dyn_ep_var[remaining_paths]

        #### update some sampler infos
        self._total_samples += alive_paths.sum()

        if self.rollout_mode=='iv_gae':
            self._total_cost += c[self.model_inds].sum()
            self._total_rew += reward[self.model_inds].sum()
            self._path_return[:,alive_paths] += reward
            self._path_cost[:,alive_paths] += c

        else:
            self._total_cost += c.sum()
            self._total_rew += reward.sum()
            self._path_return[alive_paths] += reward
            self._path_cost[alive_paths] += c

        self._path_length[alive_paths] += 1
        self._path_dyn_var[alive_paths] += np.mean(dyn_ep_var, axis=-1)
        self._total_dyn_var += dyn_ep_var.sum()

        if self.rollout_mode=='iv_gae':
            self._total_Vs += v_t[self.model_inds].sum()
            self._total_CVs += vc_t[self.model_inds].sum()
        else:
            self._total_Vs += v_t.sum()
            self._total_CVs += vc_t.sum()

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
                                        np.mean(dyn_ep_var, axis=-1),
                                        logp_t,
                                        pi_info_t,
                                        terminal)

        #### terminate mature termination due to path length
        ## update obs before finishing paths (_finish_paths() uses current obs)
        self._current_observation = next_obs

        path_end_mask = (self._path_length >= self._max_path_length-1)[alive_paths]            
        remaining_paths = self._finish_paths(term_mask=path_end_mask, append_vals=True)
        if not remaining_paths.any():
            info['alive_ratio'] = 0
            return next_obs, reward, terminal, info

        ## update remaining paths and obs
        if self.rollout_mode=='iv_gae':
            self._current_observation = self._current_observation[:,remaining_paths]
            prem_term_mask = np.any(terminal[:,remaining_paths], axis=0)            ##@anyboby maybe check later, if terminal per model should be possible
        else:
            self._current_observation = self._current_observation[remaining_paths]
            prem_term_mask = terminal
        
        #### terminate real termination due to env end
        remaining_paths = self._finish_paths(term_mask=prem_term_mask, append_vals=False)
        if not remaining_paths.any():
            info['alive_ratio'] = 0
            return next_obs, reward, terminal, info

        ### update alive paths
        alive_paths = self.pool.alive_paths
        if self.rollout_mode=='iv_gae':
            self._current_observation = self._current_observation[:,remaining_paths]
        else:
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

        # cur_obs = self._current_observation[self.model_inds]

        # We do not count env time out (mature termination) as true terminal state, append values
        if append_vals:
            if self.rollout_mode=='iv_gae':
                if self.policy.agent.reward_penalized:
                    last_val = self.policy.get_v(self._current_observation[:,term_mask], factored=False)
                else:
                    last_val = self.policy.get_v(self._current_observation[:,term_mask], factored=False)
                    last_cval = self.policy.get_vc(self._current_observation[:,term_mask], factored=False)
            else:
                if self.policy.agent.reward_penalized:
                    last_val = self.policy.get_v(self._current_observation[term_mask], factored=False)
                else:
                    last_val = self.policy.get_v(self._current_observation[term_mask], factored=False)
                    last_cval = self.policy.get_vc(self._current_observation[term_mask], factored=False)
        else:
            # init final values
            if self.rollout_mode=='iv_gae':
                last_val, last_cval = np.zeros(shape=(self.ensemble_size, term_mask.sum())), np.zeros(shape=(self.ensemble_size, term_mask.sum()))
            else:
                last_val, last_cval = np.zeros(shape=(term_mask.sum())), np.zeros(shape=(term_mask.sum()))

        self.pool.finish_path_multiple(term_mask, last_val, last_cval)

        remaining_path_mask = np.logical_not(term_mask)

        return remaining_path_mask
        
    def finish_all_paths(self):

        alive_paths=self.pool.alive_paths ##any paths that are still alive did not terminate by env
        # init final values and quantify according to termination type
        # Note: we do not count env time out as true terminal state
        if not alive_paths.any(): return self.get_diagnostics()

        if alive_paths.any():
            term_mask = np.ones(shape=alive_paths.sum(), dtype=np.bool)
            # cur_obs = self._current_observation[self.model_inds]

            if self.policy.agent.reward_penalized:
                last_val = self.policy.get_v(self._current_observation, factored=False)
            else:
                last_val = self.policy.get_v(self._current_observation, factored=False)
                last_cval = self.policy.get_vc(self._current_observation, factored=False)

            self.pool.finish_path_multiple(term_mask, last_val, last_cval)
            
        alive_paths = self.pool.alive_paths
        assert alive_paths.sum()==0   ## something went wrong with finishing all paths
        
        return self.get_diagnostics()

    def set_max_uncertainty(self, max_uncertainty):
        self.max_uncertainty = max_uncertainty