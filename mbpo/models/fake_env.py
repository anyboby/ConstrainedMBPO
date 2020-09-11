import numpy as np
import tensorflow as tf
import pdb

from mbpo.models.constructor import construct_model, format_samples_for_dyn, format_samples_for_cost
from mbpo.models.priors import WEIGHTS_PER_DOMAIN, PRIORS_BY_DOMAIN, PRIOR_DIMS, POSTS_BY_DOMAIN
from mbpo.models.utils import average_dkl, median_dkl
from mbpo.utils.logging import Progress, Silent

from itertools import count
import warnings
import time

from softlearning.policies.safe_utils.utils import discount_cumsum

EPS = 1e-8

class FakeEnv:

    def __init__(self, true_environment,
                    static_fns, num_networks=7, 
                    num_elites = 5, hidden_dims = (220, 220, 220),
                    dyn_discount = 1,
                    cost_m_discount = 1,
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
        
        self.prior_f = PRIORS_BY_DOMAIN.get(self.domain, False)
        self.post_f =  POSTS_BY_DOMAIN.get(self.domain, False)
        self.prior_dim = PRIOR_DIMS.get(self.domain, 0)
        #### create fake env from model 

        input_dim_dyn = self.obs_dim + self.prior_dim + self.act_dim
        input_dim_c = 2 * self.obs_dim + self.act_dim + self.prior_dim
        output_dim_dyn = self.active_obs_dim + self.rew_dim
        self.dyn_loss = 'NLL'

        self._model = construct_model(in_dim=input_dim_dyn, 
                                        out_dim=output_dim_dyn,
                                        name='BNN',
                                        loss=self.dyn_loss,
                                        hidden_dims=hidden_dims,
                                        lr=3e-4, 
                                        # lr_decay=0.96,
                                        # decay_steps=10000,  
                                        num_networks=num_networks, 
                                        num_elites=num_elites,
                                        weighted=dyn_discount<1,    
                                        #use_scaler=True,
                                        decay=1e-4,
                                        #sc_factor=1-1e-5,
                                        max_logvar=.5,
                                        min_logvar=-10,
                                        session=self._session)
        if self.cares_about_cost:                                                    
            
            self.cost_m_loss = 'MSE'
            
            self._cost_model = construct_model(in_dim=input_dim_c, 
                                        out_dim=1,
                                        loss=self.cost_m_loss,
                                        name='CostNN',
                                        hidden_dims=(64,64),
                                        lr=8e-5,
                                        # lr_decay=0.96,
                                        # decay_steps=10000, 
                                        num_networks=num_networks,
                                        num_elites=num_elites,
                                        weighted=cost_m_discount<1,                                            
                                        # use_scaler=True,
                                        # sc_factor=1-1e-5,
                                        # max_logvar=.5,
                                        # min_logvar=-8,
                                        decay=1e-4,
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

        if self.prior_f:
            priors = self.static_fns.prior_f(obs, act)
            inputs = np.concatenate((obs, act, priors), axis=-1)
        else:
            inputs = np.concatenate((obs, act), axis=-1)

        if obs_depth==3:
            inputs, shuffle_indxs = self.forward_shuffle(inputs)

        ensemble_dyn_means, ensemble_dyn_vars = self._model.predict(inputs, factored=True, inc_var=True)       #### outputs whole ensembles outputs

        if obs_depth==3:
            ensemble_dyn_means, ensemble_dyn_vars = self.inverse_shuffle(ensemble_dyn_means, shuffle_indxs), self.inverse_shuffle(ensemble_dyn_vars, shuffle_indxs)
        
        ensemble_dyn_means, ensemble_dyn_vars = self.filter_elite_inds(ensemble_dyn_means, self.num_elites, [ensemble_dyn_vars])
        ensemble_dyn_vars = ensemble_dyn_vars[0]
        
        ensemble_dyn_means[:,:,:-self.rew_dim] += obs           #### models output state change rather than state completely
        ensemble_model_stds = np.sqrt(ensemble_dyn_vars)
        
        ensemble_dyn_var = np.mean(ensemble_dyn_vars, axis=0)
        ensemble_dyn_var = np.mean(ensemble_dyn_var, axis=-1)

        ### calc disagreement of elites
        average_dkl_per_output = average_dkl(ensemble_dyn_means, ensemble_model_stds)
        ensemble_dkl_mean = np.mean(average_dkl_per_output, axis=tuple(np.arange(1, len(average_dkl_per_output.shape))))
        ensemble_dkl_mean = np.mean(ensemble_dkl_mean)

        median_dkl_per_output = median_dkl(ensemble_dyn_means, ensemble_model_stds)
        ensemble_dkl_med = np.mean(median_dkl_per_output, axis=tuple(np.arange(1, len(median_dkl_per_output.shape))))
        ensemble_dkl_med = np.mean(ensemble_dkl_med)
        ###

        ensemble_samples = ensemble_dyn_means

        #### choose one model from ensemble randomly
        if obs_depth<3:
            num_models, batch_size, _ = ensemble_dyn_means.shape
            model_inds = self._model.random_inds(batch_size)        ## only returns elite indices
            batch_inds = np.arange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]
        else: 
            samples = ensemble_samples

        # log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        #### retrieve r and done for new state
        next_obs = samples[...,:-self.rew_dim]

        ## post_processing
        if self.post_f:
            next_obs = self.static_fns.post_f(next_obs, act)
        
        #### ----- special steps for safety-gym ----- ####
        #### stack previous obs with newly predicted obs
        if self.safe_config:
            self.task = self.safe_config['task']
            rewards = self.static_fns.reward_f(obs, act, next_obs, self.safe_config)
            terminals = self.static_fns.termination_fn(obs, act, next_obs, self.safe_config)    ### non terminal for goal, but rebuild goal 
            next_obs = self.static_fns.rebuild_goal(obs, act, next_obs, obs, self.safe_config)  ### rebuild goal if goal was met
        #### ----- special steps for safety-gym ----- ####

        else:
            terminals = self.static_fns.termination_fn(obs, act, next_obs)
            rewards = samples[...,-self.rew_dim:]
        rew_var = np.squeeze(np.var(rewards, axis=0))

        if self.cares_about_cost:
            if self.prior_f:
                inputs_cost = np.concatenate((obs, act, next_obs, priors), axis=-1)
            else:
                inputs_cost = np.concatenate((obs, act, next_obs), axis=-1)

            costs, cost_var = self._cost_model.predict(inputs_cost, factored=False, inc_var=True)
            cost_var = (np.mean(cost_var, axis=0) + np.mean(costs**2, axis=0) - (np.mean(costs, axis=0))**2)[...,0]

        else:
            costs = np.zeros_like(rewards)
            cost_var = np.zeros(shape=rewards.shape[1:])

        # batch_size = model_means.shape[0]
        ###@anyboby TODO this calculation seems a bit suspicious to me
        # return_means = np.concatenate((model_means[:,-1:], terminals, model_means[:,:-1]), axis=-1)
        # return_stds = np.concatenate((model_stds[:,-1:], np.zeros((batch_size,1)), model_stds[:,:-1]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]
            costs = costs[0]
        
        info = {# 'return_mean': return_means,
                # 'return_std': return_stds,
                # 'log_prob': log_prob,
                # 'dev': dev,
                'dyn_ensemble_dkl_mean' : ensemble_dkl_mean,
                'dyn_ensemble_dkl_med' : ensemble_dkl_med,
                'dyn_ensemble_var_mean' : ensemble_dyn_var,
                'cost_ensemble_var' : cost_var,
                'rew_ensemble_var' : rew_var,
                'rew':rewards,
                'rew_mean': rewards.mean(),
                'cost':costs,
                'cost_mean': costs.mean(),
                }
        return next_obs, rewards, terminals, info

    def train_dyn_model(self, samples, discount=1, **kwargs):
        # check priors
        priors = self.static_fns.prior_f(samples['observations'], samples['actions']) if self.prior_f else None

        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
        if discount<1:
            train_inputs_dyn, train_outputs_dyn, weights = format_samples_for_dyn(samples, 
                                                                        priors=priors,
                                                                        safe_config=self.safe_config,
                                                                        discount=discount,
                                                                        #noise=1e-4
                                                                        )
            kwargs['weights'] = weights
            model_metrics = self._model.train(train_inputs_dyn, 
                                                train_outputs_dyn, 
                                                **kwargs,
                                                )
        
        else:
            train_inputs_dyn, train_outputs_dyn = format_samples_for_dyn(samples, 
                                                                        priors=priors,
                                                                        safe_config=self.safe_config,
                                                                        #noise=1e-4
                                                                        )
            
            model_metrics = self._model.train(train_inputs_dyn, 
                                                train_outputs_dyn, 
                                                **kwargs,
                                                )


        return model_metrics

    def train_cost_model(self, samples, discount=1, **kwargs):        
        # check priors
        priors = self.static_fns.prior_f(samples['observations'], samples['actions']) if self.prior_f else None
        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
        
        if discount<1:
            inputs, targets, weights = format_samples_for_cost(samples, 
                                                        one_hot=False,
                                                        priors=priors,
                                                        discount=discount,
                                                        # noise=1e-4
                                                        )
            kwargs['weights'] = weights
            cost_model_metrics = self._cost_model.train(inputs,
                                        targets,
                                        **kwargs,
                                        )                                            
        else:
            inputs, targets = format_samples_for_cost(samples, 
                                                        one_hot=False,
                                                        priors=priors,
                                                        # noise=1e-4
                                                        )
            #### Useful Debugger line: np.where(np.max(train_inputs_cost[np.where(train_outputs_cost[:,1]>0.8)][:,3:54], axis=1)<0.95)

            cost_model_metrics = self._cost_model.train(inputs,
                                        targets,
                                        **kwargs,
                                        )                                            
            
        return cost_model_metrics

    def random_inds(self, size):
        return self._model.random_inds(batch_size=size)

    def reset_model(self):
        self._model.reset()
        self._cost_model.reset()
        
    def filter_elite_inds(self, data, n_elites, apply_too = None):
        '''
        extracts the closest data to the median
        data 0-axis is ensemble axis
        data 1-axis is batch axis
        apply_too: a list of arrays with same dims as data that the same filtration is applied to. 
        '''
        ### swap for convenience
        data_sw = np.swapaxes(data, 0, 1)
        mse_median = np.mean((data_sw-np.median(data_sw, axis=1)[:,None,...])**2, axis=-1)
        sorted_inds = np.argsort(mse_median, axis=1)[:, :n_elites]
        replace_inds = sorted_inds[:, 0:self.num_networks-n_elites]
        batch_inds = np.arange(data_sw.shape[0])[...,None]

        res = np.concatenate((data_sw[batch_inds, sorted_inds], data_sw[batch_inds, replace_inds]), axis=1)
        res = np.swapaxes(res, 0,1)

        if apply_too is not None:
            sw_list = [np.swapaxes(arr, 0, 1) for arr in apply_too]
            res_list_too = [np.concatenate((sw_too[batch_inds, sorted_inds], sw_too[batch_inds, replace_inds]), axis=1) for sw_too in sw_list]
            res_list_too = [np.swapaxes(res_too, 0,1) for res_too in res_list_too]
            return res, res_list_too
        return res

    def forward_shuffle(self, ndarray):
        """
        shuffles ndarray forward along axis 0 with random elite indices, 
        Returns shuffled copy of ndarray and indices with which was shuffled
        """
        idxs = np.random.permutation(ndarray.shape[0])
        shuffled = ndarray[idxs]
        return shuffled, idxs

    def inverse_shuffle(self, ndarray, idxs):
        """
        inverses a shuffle of ndarray forward along axis 0, given the used indices. 
        Returns unshuffled copy of ndarray
        """
        unshuffled = ndarray[idxs]
        return unshuffled


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
