import numpy as np
import tensorflow as tf
import pdb

from mbpo.models.constructor import construct_model, format_samples_for_dyn, format_samples_for_cost
from mbpo.models.priors import WEIGHTS_PER_DOMAIN, PRIORS_BY_DOMAIN, PRIOR_DIMS, POSTS_BY_DOMAIN
from mbpo.models.utils import average_dkl
from mbpo.utils.logging import Progress, Silent

from itertools import count
import warnings
import time

from softlearning.policies.safe_utils.utils import discount_cumsum

EPS = 1e-8

class FakeEnv:

    def __init__(self, true_environment, policy,
                    static_fns, num_networks=7, 
                    num_elites = 5, hidden_dims = (220, 220, 220),
                    cares_about_cost=False, 
                    safe_config=None,
                    session = None):
        
        self.domain = true_environment.domain
        self.env = true_environment
        self.obs_dim = np.prod(self.observation_space.shape)
        self.act_dim = np.prod(self.action_space.shape)
        self.active_obs_dim = int(self.obs_dim/self.env.stacks)
        self._session = session
        self.cares_about_cost = cares_about_cost
        self.rew_dim = 1
        self.cost_classes = [0,1]

        self.num_networks = num_networks
        self.num_elites = num_elites

        self.policy = policy

        self.static_fns = static_fns
        self.safe_config = safe_config
        if safe_config:
            self.stacks = self.safe_config['stacks']
            self.stacking_axis = self.safe_config['stacking_axis']
        else:
            self.stacks = 1
            self.stacking_axis = 0

        target_weight_f = WEIGHTS_PER_DOMAIN.get(self.domain, None)
        self.target_weights = target_weight_f(self.obs_dim) if target_weight_f else None
        # self.prior_f = PRIORS_BY_DOMAIN.get(self.domain, None)
        # self.post_f = POSTS_BY_DOMAIN.get(self.domain, None)
        
        self.prior_f = True
        self.post_f = True
        self.prior_dim = PRIOR_DIMS.get(self.domain, 0)
        #### create fake env from model 

        input_dim_dyn = self.obs_dim + self.prior_dim + self.act_dim
        input_dim_c = self.obs_dim + self.prior_dim
        output_dim_dyn = self.active_obs_dim + self.rew_dim
        self._model = construct_model(in_dim=input_dim_dyn, 
                                        out_dim=output_dim_dyn,
                                        name='BNN',
                                        hidden_dims=hidden_dims,
                                        lr=3e-4, 
                                        # lr_decay=0.96,
                                        # decay_steps=10000,  
                                        num_networks=num_networks, 
                                        num_elites=num_elites,
                                        weights=self.target_weights,    
                                        use_scaler=True,
                                        sc_factor=.99,
                                        max_logvar=-1,
                                        min_logvar=-10,
                                        session=self._session)

        if self.cares_about_cost:                                                    
            self._cost_model = construct_model(in_dim=input_dim_c, 
                                        out_dim=1,
                                        loss='MSE',
                                        name='CostNN',
                                        hidden_dims=(128, 128, 128),
                                        lr=7e-5, 
                                        # lr_decay=0.96,
                                        # decay_steps=10000, 
                                        num_networks=num_networks,
                                        num_elites=num_elites,
                                        use_scaler=True,
                                        sc_factor=.99,
                                        session=self._session)
            
        else:
            self._cost_model = None
        
        self._static_fns = static_fns           # termination functions for the envs (model can't simulate those)
        self.running_mean_stdscale = 1
        self.last_tic = time.perf_counter()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        obs_depth = len(obs.shape)
        if obs_depth == 1:
            obs = obs[None]
            act = act[None]
            return_single=True
        else:
            return_single = False

        unstacked_obs_size = int(obs.shape[1+self.stacking_axis]/self.stacks)               ### e.g. if a stacked obs is 88 with 4 stacks,
                                                                                            ### unstacking it yields 22
        if self.prior_f:
            priors = self.static_fns.prior_f(obs, act)
            inputs = np.concatenate((obs, act, priors), axis=-1)
        else:
            inputs = np.concatenate((obs, act), axis=-1)

        ensemble_model_means, ensemble_model_vars = self._model.predict(inputs, factored=True)       #### self.model outputs whole ensembles outputs

        ####@anyboby TODO only as an example, do a better disagreement measurement later
        # ensemble_disagreement_means = np.nanvar(ensemble_model_means, axis=0)*self.target_weights
        # ensemble_disagreement_stds = np.sqrt(np.var(ensemble_model_vars, axis=0))*self.target_weights
        # ensemble_disagreement_logstds = np.log(ensemble_disagreement_stds)
        # ensemble_disagreement = np.sum(ensemble_disagreement_means+ensemble_disagreement_stds, axis=-1)

        ensemble_model_means[:,:,:-self.rew_dim] += obs[:,-unstacked_obs_size:]           #### models output state change rather than state completely
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        mean_ensemble_var = ensemble_model_vars.mean()

        #### check for negative vars (can happen towards end of training)
        if (ensemble_model_vars<=0).any():
            neg_var = ensemble_model_vars.min()
            np.clip(ensemble_model_vars, a_min=1e-8, a_max=None)
            warnings.warn(f'Negative variance of {neg_var} encountered. Clipping...')

        ### calc disagreement of elites
        elite_means = ensemble_model_means[self._model.elite_inds]
        elite_stds = ensemble_model_stds[self._model.elite_inds]
        average_dkl_per_output = average_dkl(elite_means, elite_stds)#*self.target_weights
        average_dkl_mean = np.mean(average_dkl_per_output, axis=tuple(np.arange(1, len(average_dkl_per_output.shape))))
        ensemble_disagreement = average_dkl_mean

        ### directly use means, if deterministic
        if deterministic:   
            ensemble_samples = ensemble_model_means                     
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        #### choose one model from ensemble randomly
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = self._model.random_inds(batch_size)        ## only returns elite indices
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[model_inds, batch_inds]
        model_means = ensemble_model_means[model_inds, batch_inds]
        model_stds = ensemble_model_stds[model_inds, batch_inds]
        ####

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        #### retrieve r and done for new state
        rewards, next_obs = samples[:,-self.rew_dim:], samples[:,:-self.rew_dim]

        ## post_processing
        if self.post_f:
            next_obs = self.static_fns.post_f(next_obs, act)
        
        #### ----- special steps for safety-gym ----- ####
        #### stack previous obs with newly predicted obs
        if self.safe_config:
            self.task = self.safe_config['task']
            unstacked_obs = obs[:,-unstacked_obs_size:]
            rewards = self.static_fns.reward_f(unstacked_obs, act, next_obs, self.safe_config)
            terminals = self.static_fns.termination_fn(unstacked_obs, act, next_obs, self.safe_config)    ### non terminal for goal, but rebuild goal 
            next_obs = self.static_fns.rebuild_goal(unstacked_obs, act, next_obs, unstacked_obs, self.safe_config)  ### rebuild goal if goal was met
            if self.stacks > 1:
                next_obs = np.concatenate((obs, next_obs), axis=-((obs_depth-1)-self.stacking_axis))
                next_obs = np.delete(next_obs, slice(unstacked_obs_size), -((obs_depth-1)-self.stacking_axis))
        #### ----- special steps for safety-gym ----- ####
        else:
            terminals = self.static_fns.termination_fn(obs, act, next_obs)

        if self.cares_about_cost:
            if self.prior_f:
                inputs_cost = np.concatenate((next_obs, priors), axis=-1)
            else:
                inputs_cost = next_obs

            costs = self._cost_model.predict(inputs_cost, factored=True)
            # costs = np.random.normal(size=costs.shape) * np.sqrt(costs_var)
            costs = np.squeeze(np.clip(costs, -1, 1))
            ensemble_disagreement = np.var(costs, axis=0)
            costs = np.mean(costs, axis=0)

        else:
            costs = np.zeros_like(rewards)

        batch_size = model_means.shape[0]
        ###@anyboby TODO this calculation seems a bit suspicious to me
        return_means = np.concatenate((model_means[:,-1:], terminals, model_means[:,:-1]), axis=-1)
        return_stds = np.concatenate((model_stds[:,-1:], np.zeros((batch_size,1)), model_stds[:,:-1]), axis=-1)

        ### save state and action
        self._current_obs = next_obs
        self._last_act = act

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]
            costs = costs[0]
        
        info = {'return_mean': return_means,
                'return_std': return_stds,
                'log_prob': log_prob,
                'dev': dev,
                'ensemble_disagreement':ensemble_disagreement,
                'ensemble_var':mean_ensemble_var,
                'cost':costs,
                'cost_mean': costs.mean(),
                }
        return next_obs, rewards, terminals, info


    def invVarRollout(self, obs, gamma=0.99, c_gamma=0.99, lam=0.97, c_lam=0.97 ,horizon=100, stop_var = 1e3):
        if len(obs.shape)==1:
            obs = obs[None,None]
            obs = np.repeat(obs, repeats=self.num_networks, axis=0)
        elif len(obs.shape)==2:
            obs = obs[None]
            obs = np.repeat(obs, repeats=self.num_networks, axis=0)
        assert len(obs.shape)==3
        batch_size = len(obs[0])

        alive_mask = np.ones(shape=(batch_size), dtype=np.bool)
        population_mask = np.zeros(shape=(batch_size, horizon), dtype=np.bool)

        rew_buf = np.zeros(shape=(self.num_networks, batch_size, horizon), dtype=np.float32)

        ret_buf = np.zeros(shape=(batch_size, horizon), dtype=np.float32)
        ret_buf_med = np.zeros(shape=(batch_size, horizon), dtype=np.float32)

        ret_var_buf = np.zeros(shape=(batch_size, horizon), dtype=np.float32)
        ret_iod2_buf = np.zeros(shape=(batch_size, horizon), dtype=np.float32)

        cost_buf = np.zeros(shape=(self.num_networks, batch_size, horizon), dtype=np.float32)
        cost_buf_cl = np.zeros(shape=(self.num_networks, batch_size, horizon), dtype=np.float32)

        cret_buf = np.zeros(shape=(batch_size, horizon), dtype=np.float32)
        cret_buf_mean = np.zeros(shape=(batch_size, horizon), dtype=np.float32)
        cret_buf_median_cl = np.zeros(shape=(batch_size, horizon), dtype=np.float32)
        cret_buf_mean_cl = np.zeros(shape=(batch_size, horizon), dtype=np.float32)

        cret_var_buf = np.zeros(shape=(batch_size, horizon), dtype=np.float32)
        cret_iod_buf = np.zeros(shape=(batch_size, horizon), dtype=np.float32)
        cret_iod2_buf = np.zeros(shape=(batch_size, horizon), dtype=np.float32)
        cret_iod4_buf = np.zeros(shape=(batch_size, horizon), dtype=np.float32)

        tic = time.perf_counter()
        
        cvals = self.policy.get_vc(obs, inc_var=False)
        cvals = np.mean(cvals, axis=0)

        vals = self.policy.get_v(obs, inc_var=False)
        vals = np.mean(vals, axis=0)

        self.tic(verbose=False) #reset timer
        t0 = time.time()
        progress = Progress(horizon)

        for i in range(horizon):
            if alive_mask.sum()<5: break
            alive_obs = obs[:, alive_mask, :]
            get_action_outs = self.policy.get_action_outs(alive_obs, inc_v = False)

            a = get_action_outs['pi']

            logp_t = get_action_outs['logp_pi']
            pi_info_t = get_action_outs['pi_info']
        
            if self.prior_f:
                priors = self.static_fns.prior_f(alive_obs, a)

                inputs = np.concatenate((alive_obs, a, priors), axis=-1)
            else:
                inputs = np.concatenate((alive_obs, a), axis=-1)

            ens_means, ens_vars = self._model.predict(inputs)

            ens_means[...,:-self.rew_dim] += alive_obs         #### models output state change rather than state completely

            # @TODO anyboby
            # super slow, maybe revive later
            # ens_stds = np.sqrt(ens_vars)
            # ens_samples = ens_means + np.random.normal(size=ens_means.shape) * ens_stds
            
            ens_samples = ens_means

            #### retrieve r and done for new state
            rew, next_obs = ens_samples[...,-self.rew_dim:], ens_samples[...,:-self.rew_dim]

            ## post_processing
            if self.post_f:
                next_obs = self.static_fns.post_f(next_obs, a)
            
            if self.safe_config:
                self.task = self.safe_config['task']
                
                rew = np.squeeze(self.static_fns.reward_f(alive_obs, a, next_obs, self.safe_config))

                terminals = np.squeeze(self.static_fns.termination_fn(alive_obs, a, next_obs, self.safe_config))    ### non terminal for goal, but rebuild goal 

                next_obs = self.static_fns.rebuild_goal(alive_obs, a, next_obs, obs, self.safe_config)  ### rebuild goal if goal was met

            else:
                terminals = np.squeeze(self.static_fns.termination_fn(alive_obs, a, next_obs))

            if self.cares_about_cost:
                if self.prior_f:
                    inputs_cost = np.concatenate((next_obs, priors), axis=-1)
                else:
                    inputs_cost = next_obs
                
                costs = np.squeeze(self._cost_model.predict(inputs_cost, factored=True))
                costs_cl = np.clip(costs, 0, 1)
                

            else:
                costs = np.zeros_like(rew)

            ## we only care about start state, can just discount in the forward sim
            rew_buf[:, alive_mask, i] = rew*gamma**i
            cost_buf[:, alive_mask, i] = costs*c_gamma**i
            cost_buf_cl[:, alive_mask, i] = costs_cl*c_gamma**i

            ###### DEBUg ############
            # costs_deb = np.squeeze(np.mean(costs, axis=0)[0:5])
            # costs_real = real_samples[9][(rand_inds+i)[0:5]]
            # dc = (costs_deb-costs_real)**2
            # obs_deb = np.mean(obs[:,0:5,:], axis=0)
            # obs_real = real_samples[0][(rand_inds+i)[0:5]]
            # d_obs = (obs_deb-obs_real)**2
            #########################

            next_val = self.policy.get_v(next_obs)
            next_cval = self.policy.get_vc(next_obs)

            next_val *= np.logical_not(terminals)
            next_cval *= np.logical_not(terminals)

            # next_val_var *= np.logical_not(terminals)
            # next_cval_var *= np.logical_not(terminals)

            #### variance for gaussian mixture, add dispersion of means to variance
            # next_val_var = np.mean(next_cval_var, axis=0) + np.mean(next_val**2, axis=0) - (np.mean(next_val, axis=0))**2
            # next_cval_var = np.mean(next_cval_var, axis=0) + np.mean(next_cval**2, axis=0) - (np.mean(next_cval, axis=0))**2
            next_val_var = np.var(next_val, axis=0)
            next_cval_var = np.var(next_cval, axis=0)

            ####

            disc_ret = np.sum(rew_buf[:, alive_mask, 0:i+1], axis=-1)
            disc_cret = np.sum(cost_buf[:, alive_mask, 0:i+1], axis=-1)
            disc_cret_cl = np.sum(cost_buf_cl[:, alive_mask, 0:i+1], axis=-1)

            # self.tic('discumsum')

            bootstrap_v = gamma**i*next_val
            bootstrap_v_var = gamma**i*next_val_var
            bootstrap_vc = c_gamma**i*next_cval
            bootstrap_vc_var = c_gamma**i*next_cval_var

            ret_buf[alive_mask, i] = np.mean(disc_ret+bootstrap_v, axis=0)
            ret_buf_med[alive_mask, i] = np.median(disc_ret+bootstrap_v, axis=0)
            
            ret_var_buf[alive_mask, i] = (np.var(disc_ret, axis=0) + bootstrap_v_var)/(lam**i)
            ret_iod2_buf[alive_mask, i] = ret_var_buf[alive_mask, i]/np.square(ret_buf[alive_mask, i])

            cret_buf[alive_mask, i] = np.median(disc_cret+bootstrap_vc, axis=0)
            cret_buf_mean[alive_mask, i] = np.mean(disc_cret+bootstrap_vc, axis=0)
            cret_buf_median_cl[alive_mask, i] = np.median(disc_cret_cl+bootstrap_vc, axis=0)
            cret_buf_mean_cl[alive_mask, i] = np.mean(disc_cret_cl+bootstrap_vc, axis=0)

            cret_var_buf[alive_mask, i] = (np.var(disc_cret, axis=0) + bootstrap_vc_var)/(c_lam**i)
            cret_iod_buf[alive_mask, i] = cret_var_buf[alive_mask, i]/np.abs(cret_buf[alive_mask, i])
            cret_iod2_buf[alive_mask, i] = cret_var_buf[alive_mask, i]/np.square(cret_buf[alive_mask, i])
            cret_iod4_buf[alive_mask, i] = np.square(cret_var_buf[alive_mask, i]/np.square(cret_buf[alive_mask, i]))

            # self.tic('buffers')
            #### rearrange next obs to remove outliers
            no_sw = np.swapaxes(next_obs, 0, 1)
            cret_sw = np.swapaxes(disc_cret, 0, 1)
            elite_idx4 = np.squeeze(np.argsort(cret_sw-np.mean(cret_sw, axis=1)[...,None], axis=1))[...,:4]
            elite_idx3 = np.squeeze(np.argsort(cret_sw-np.mean(cret_sw, axis=1)[...,None], axis=1))[...,:3]
            batch_idx = np.arange(len(elite_idx4))[...,None]
            next_obs = np.concatenate((no_sw[batch_idx, elite_idx4], no_sw[batch_idx, elite_idx3]), axis=1)
            next_obs = np.swapaxes(next_obs, 0, 1)

            # self.tic('elite obs')
            progress.set_description([['Rollout Number',f'{i+1}']] + [['T', time.time() - t0]])
            progress.update()
            obs[:, alive_mask, :] = next_obs
            population_mask[alive_mask, i] = True
            alive_mask *= np.squeeze(np.logical_and(ret_var_buf[:, i]<stop_var, cret_var_buf[:, i]<stop_var))

        
        wr = np.where(population_mask, 1/(ret_var_buf+EPS), 0)
        wr[(wr-np.mean(wr, axis=-1)[...,None])/(np.sqrt(np.var(wr, axis=-1))+EPS)[...,None]<0]=0        ### normalize and remove outliers
        wr_sum = np.sum(wr, axis=-1)[...,None]

        wr_iod2 = np.where(population_mask, 1/(ret_iod2_buf+EPS), 0)
        wr_iod2[(wr_iod2-np.mean(wr_iod2, axis=-1)[...,None])/(np.sqrt(np.var(wr_iod2, axis=-1))+EPS)[...,None]<0]=0        ### normalize and remove outliers
        wr_iod2_sum = np.sum(wr_iod2, axis=-1)[...,None]

        returns = np.sum(ret_buf*wr/wr_sum, axis=-1)
        returns_med_iod2 = np.sum(ret_buf_med*wr_iod2*1/(wr_iod2_sum+EPS), axis=-1)

        a_rew = returns - vals
        a_rew = (a_rew-np.mean(a_rew))/(np.sqrt(np.var(a_rew))+EPS)

        a_rew_iod2 = returns_med_iod2 - vals
        a_rew_iod2 = (a_rew_iod2-np.mean(a_rew_iod2))/(np.sqrt(np.var(a_rew_iod2))+EPS)

        a_rew_td0 = ret_buf[:,0] - cvals
        a_rew_td0 = (a_rew_td0-np.mean(a_rew_td0))/(np.sqrt(np.var(a_rew_td0))+EPS)

        # e_rew_1 = np.mean((returns-real_samples[4][rand_inds])**2)
        # e_rew_2 = np.mean((returns_med_iod2-real_samples[4][rand_inds])**2)
        # e_rew_td0 = np.mean((ret_buf[:,0]-real_samples[4][rand_inds])**2)

        # ea_rew_1 = np.mean((a_rew-real_samples[2][rand_inds])**2)
        # ea_rew_2 = np.mean((a_rew_iod2-real_samples[2][rand_inds])**2)
        # ea_rew_td0 = np.mean((a_rew_td0-real_samples[2][rand_inds])**2)
        
        ##############
        #### Cost ####
        ##############

        wc = np.where(population_mask, 1/(cret_var_buf+EPS), 0)
        wc[(wc-np.mean(wc, axis=-1)[..., None])/(np.sqrt(np.var(wc, axis=-1))+EPS)[..., None]<0]=0
        wc_sum = np.sum(wc, axis=-1)[..., None]

        wc_iod = np.where(population_mask, 1/(cret_iod_buf+EPS), 0)
        wc_iod[(wc_iod-np.mean(wc_iod, axis=-1)[..., None])/(np.sqrt(np.var(wc_iod, axis=-1))+EPS)[..., None]<0]=0
        wc_sum_iod = np.sum(wc_iod, axis=-1)[..., None]
        
        wc_iod2 = np.where(population_mask, 1/(cret_iod2_buf+EPS), 0)
        wc_iod2[(wc_iod2-np.mean(wc_iod2, axis=-1)[..., None])/(np.sqrt(np.var(wc_iod2, axis=-1))+EPS)[..., None]<0]=0
        wc_sum_iod2 = np.sum(wc_iod2, axis=-1)[..., None]

        wc_iod4 = np.where(population_mask, 1/(cret_iod4_buf+EPS), 0)
        wc_iod4[(wc_iod4-np.mean(wc_iod4, axis=-1)[..., None])/(np.sqrt(np.var(wc_iod4, axis=-1))+EPS)[..., None]<0]=0
        wc_sum_iod4 = np.sum(wc_iod4, axis=-1)[..., None]

        creturns_med = np.sum(cret_buf*wc*1/wc_sum, axis=-1)
        creturns_med_iod = np.sum(cret_buf*wc_iod*1/wc_sum_iod, axis=-1)
        creturns_mean = np.sum(cret_buf_mean*wc*1/wc_sum, axis=-1)
        creturns_mean_iod = np.sum(cret_buf_mean*wc_iod*1/wc_sum_iod, axis=-1)
        creturns_med_cl = np.sum(cret_buf_median_cl*wc*1/wc_sum, axis=-1)
        creturns_mean_cl = np.sum(cret_buf_mean_cl*wc*1/wc_sum, axis=-1)
        creturns_med_cl_iod = np.sum(cret_buf_median_cl*wc_iod*1/wc_sum_iod, axis=-1)
        creturns_mean_cl_iod = np.sum(cret_buf_mean_cl*wc_iod*1/wc_sum_iod, axis=-1)
        creturns_med_cl_iod2 = np.sum(cret_buf_median_cl*wc_iod2*1/wc_sum_iod2, axis=-1)
        creturns_med_cl_iod4 = np.sum(cret_buf_median_cl*wc_iod4*1/wc_sum_iod4, axis=-1)

        a_med = creturns_med - cvals
        a_med -= np.mean(a_med)

        a_med_iod = creturns_med_iod - cvals
        a_med_iod -= np.mean(a_med_iod)

        a_mean = creturns_mean - cvals
        a_mean -= np.mean(a_mean)

        a_mean_iod = creturns_mean_iod - cvals
        a_mean_iod -= np.mean(a_mean_iod)

        a_med_cl = creturns_med_cl - cvals
        a_med_cl -= np.mean(a_med_cl)

        a_mean_cl = creturns_mean_cl - cvals
        a_mean_cl -= np.mean(a_mean_cl)
        
        a_med_cl_iod = creturns_med_cl_iod - cvals
        a_med_cl_iod -= np.mean(a_med_cl_iod)

        a_mean_cl_iod = creturns_mean_cl_iod - cvals
        a_mean_cl_iod -= np.mean(a_mean_cl_iod)

        a_med_cl_iod2 = creturns_med_cl_iod2 - cvals
        a_med_cl_iod2 -= np.mean(a_med_cl_iod2)
        a_med_cl_iod2 *= self.running_mean_stdscale
        
        a_med_cl_iod4 = creturns_med_cl_iod4 - cvals
        a_med_cl_iod4 -= np.mean(a_med_cl_iod4)
        a_med_cl_iod4 *= self.running_mean_stdscale

        a_td0 = cret_buf[:, 0] - cvals
        a_td0 -= np.mean(a_td0)

        # self.running_mean_stdscale += 0.1 * (np.sqrt(np.var(real_samples[3][rand_inds]))/np.sqrt(np.var(a_med))-self.running_mean_stdscale)
        # a_med_resc = a_med * self.running_mean_stdscale

        # e1 = np.mean((creturns_med-real_samples[5][rand_inds])**2)
        # e2 = np.mean((creturns_med_iod-real_samples[5][rand_inds])**2)
        # e3 = np.mean((creturns_mean-real_samples[5][rand_inds])**2)
        # e4 = np.mean((creturns_mean_iod-real_samples[5][rand_inds])**2)
        # e5 = np.mean((creturns_med_cl-real_samples[5][rand_inds])**2)
        # e6 = np.mean((creturns_mean_cl-real_samples[5][rand_inds])**2)
        # e7 = np.mean((creturns_med_cl_iod-real_samples[5][rand_inds])**2)
        # e8 = np.mean((creturns_mean_cl_iod-real_samples[5][rand_inds])**2)
        # e9 = np.mean((creturns_med_cl_iod2-real_samples[5][rand_inds])**2)
        # e9 = np.mean((creturns_med_cl_iod4-real_samples[5][rand_inds])**2)
        # etd0 = np.mean((cret_buf[:, 0]-real_samples[5][rand_inds])**2)

        # ea1 = np.mean((a_med-real_samples[3][rand_inds])**2)
        # ea1_resc = np.mean((a_med_resc-real_samples[3][rand_inds])**2)
        # ea2 = np.mean((a_med_iod-real_samples[3][rand_inds])**2)
        # ea3 = np.mean((a_mean-real_samples[3][rand_inds])**2)
        # ea4 = np.mean((a_mean_iod-real_samples[3][rand_inds])**2)
        # ea5 = np.mean((a_med_cl-real_samples[3][rand_inds])**2)
        # ea6 = np.mean((a_mean_cl-real_samples[3][rand_inds])**2)
        # ea7 = np.mean((a_med_cl_iod-real_samples[3][rand_inds])**2)
        # ea8 = np.mean((a_mean_cl_iod-real_samples[3][rand_inds])**2)
        # ea9 = np.mean((a_med_cl_iod2-real_samples[3][rand_inds])**2)
        # ea10 = np.mean((a_med_cl_iod4-real_samples[3][rand_inds])**2)
        # eatd0 = np.mean((a_td0-real_samples[3][rand_inds])**2)

        #creturns_med = creturns_med - np.mean(creturns_med)
        ### diag
        rolls = np.tile(np.arange(horizon)[None, ...], reps=(wr.shape[0], 1))
        r_H_mean = np.mean(np.sum(rolls*wr/wr_sum, axis=-1))
        c_H_mean = np.mean(np.sum(rolls*wc/wc_sum, axis=-1))
        diagnostics = dict(r_H_mean=r_H_mean, c_H_mean=c_H_mean)
        print(f'Inverse Variance Rollout Finished')
        print(f'average return horizon: {r_H_mean}, average cost horizon: {c_H_mean}')
        
        return returns, creturns_med, a_rew, a_med_cl, diagnostics

    def train_dyn_model(self, samples, **kwargs):
        # check priors
        priors = self.static_fns.prior_f(samples['observations'], samples['actions']) if self.prior_f else None

        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
        train_inputs_dyn, train_outputs_dyn = format_samples_for_dyn(samples, 
                                                                    priors=priors,
                                                                    safe_config=self.safe_config,
                                                                    noise=1e-4)
        
        model_metrics = self._model.train(train_inputs_dyn, 
                                            train_outputs_dyn, 
                                            **kwargs,
                                            )
        
        return model_metrics

    def train_cost_model(self, samples, **kwargs):        
        # check priors
        priors = self.static_fns.prior_f(samples['observations'], samples['actions']) if self.prior_f else None
        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
        inputs, targets = format_samples_for_cost(samples, 
                                                    one_hot=False,
                                                    priors=priors,
                                                    noise=1e-4)
        #### Useful Debugger line: np.where(np.max(train_inputs_cost[np.where(train_outputs_cost[:,1]>0.8)][:,3:54], axis=1)<0.95)

        cost_model_metrics = self._cost_model.train(inputs,
                                    targets,
                                    **kwargs,
                                    )                                            
        return cost_model_metrics


    def reset_model(self):
        self._model.reset()
        self._cost_model.reset()
        

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,-1:], ensemble_model_means[:,:,:-1] + obs_ph[None]], axis=-1)
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        samples = ensemble_samples[0]

        rewards, next_obs = samples[:,-1:], samples[:,:-1]
        terminals = self.static_fns.termination_ph_fn(obs_ph, act_ph, next_obs)
        info = {}

        return next_obs, rewards, terminals, info

    def close(self):
        pass

    def _random_choice_prob_index(self, a, axis=1):
        r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
        return (a.cumsum(axis=axis) > r).argmax(axis=axis)

    def tic(self, message='time since last tic', verbose=True):
        tic = time.perf_counter()
        if verbose:
            print(message,tic-self.last_tic)
        self.last_tic = tic
