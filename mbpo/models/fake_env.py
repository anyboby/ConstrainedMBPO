import numpy as np
import tensorflow as tf
import pdb

from mbpo.models.constructor import construct_model, format_samples_for_dyn, format_samples_for_cost
from mbpo.models.priors import WEIGHTS_PER_DOMAIN, PRIORS_BY_DOMAIN, PRIOR_DIMS, POSTS_BY_DOMAIN
from mbpo.models.utils import average_dkl

import warnings

class FakeEnv:

    def __init__(self, true_environment, 
                    static_fns, num_networks=7, 
                    num_elites = 5, hidden_dims = (220, 220, 220),
                    cares_about_cost=False, 
                    safe_config=None,
                    session = None):
        
        self.domain = true_environment.domain
        self.env = true_environment
        self.obs_dim = np.prod(self.env.observation_space.shape)
        self.act_dim = np.prod(self.env.action_space.shape)
        self.active_obs_dim = int(self.obs_dim/self.env.stacks)
        self._session = session
        self.cares_about_cost = cares_about_cost
        self.rew_dim = 1
        self.cost_classes = [0,1]

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
        self.prior_f = PRIORS_BY_DOMAIN.get(self.domain, None)
        self.post_f = POSTS_BY_DOMAIN.get(self.domain, None)
        prior_dim = PRIOR_DIMS.get(self.domain, 0)
        #### create fake env from model 

        input_dim_dyn = self.obs_dim + prior_dim + self.act_dim
        input_dim_c = self.obs_dim + prior_dim
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
                                        session=self._session)

        if self.cares_about_cost:
            self._cost_model = construct_model(in_dim=input_dim_c, 
                                        out_dim=2,
                                        loss='CE',
                                        name='CostNN',
                                        hidden_dims=(128, 128, 128),
                                        output_activation='softmax',
                                        lr=3e-5, 
                                        # lr_decay=0.96,
                                        # decay_steps=10000, 
                                        num_networks=num_networks, 
                                        num_elites=num_elites,
                                        session=self._session)
        else:
            self._cost_model = None
        
        self._static_fns = static_fns           # termination functions for the envs (model can't simulate those)


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
            priors = self.prior_f(obs, act)
            inputs = np.concatenate((obs, act, priors), axis=-1)
            inputs_cost = np.concatenate((obs, priors), axis=-1)
        else:
            inputs = np.concatenate((obs, act), axis=-1)
            inputs_cost = obs

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
        average_dkl_per_output = average_dkl(elite_means, elite_stds)*self.target_weights
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

        if self.cares_about_cost:
            cost_prob = self._cost_model.predict(inputs_cost, factored=True)
            cost_model_inds = self._cost_model.random_inds(batch_size) 
            cost_prob = cost_prob[cost_model_inds, batch_inds]
            cost_batch = np.tile(self.cost_classes, (batch_size,1))
            costs = cost_batch[(batch_inds, self._random_choice_prob_index(cost_prob, axis=1))]  ### chooses indices by p-dist
        else:
            costs = np.zeros_like(rewards)

        #### ----- special steps for safety-gym ----- ####
        #### stack previous obs with newly predicted obs
        if self.safe_config:
            self.task = self.safe_config['task']
            unstacked_obs = obs[:,-unstacked_obs_size:]
            rewards = self.static_fns.reward_np(unstacked_obs, act, next_obs, self.safe_config)
            terminals = self.static_fns.termination_fn(unstacked_obs, act, next_obs, self.safe_config)    ### non terminal for goal, but rebuild goal 
            next_obs = self.static_fns.rebuild_goal(unstacked_obs, act, next_obs, unstacked_obs, self.safe_config)  ### rebuild goal if goal was met
            if self.stacks > 1:
                next_obs = np.concatenate((obs, next_obs), axis=-((obs_depth-1)-self.stacking_axis))
                next_obs = np.delete(next_obs, slice(unstacked_obs_size), -((obs_depth-1)-self.stacking_axis))
        #### ----- special steps for safety-gym ----- ####
        else:
            terminals = self.static_fns.termination_fn(obs, act, next_obs)

        ## post_processing
        if self.post_f:
            next_obs = self.post_f(next_obs, act)

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

    def train_dyn_model(self, samples, **kwargs):        
        # check priors
        priors = self.prior_f(samples['observations'], samples['actions']) if self.prior_f else None

        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
        train_inputs_dyn, train_outputs_dyn = format_samples_for_dyn(samples, 
                                                                    priors=priors,
                                                                    safe_config=self.safe_config,
                                                                    noise=5e-4)
        
        model_metrics = self._model.train(train_inputs_dyn, 
                                            train_outputs_dyn, 
                                            **kwargs,
                                            )
        
        return model_metrics

    def train_cost_model(self, samples, **kwargs):        
        # check priors
        priors = self.prior_f(samples['observations'], samples['actions']) if self.prior_f else None

        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
        train_inputs_cost, train_outputs_cost = format_samples_for_cost(samples, 
                                                                    one_hot=True,
                                                                    num_classes=len(self.cost_classes),
                                                                    priors=priors,
                                                                    noise=5e-3)
        model_metrics_cost = self._cost_model.train(train_inputs_cost,
                                    train_outputs_cost,
                                    **kwargs,
                                    )                                            
        
        return model_metrics_cost


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

