import numpy as np
import tensorflow as tf
import pdb

from mbpo.models.constructor import construct_model, format_samples_for_training
from mbpo.models.priors import WEIGHTS_PER_DOMAIN, PRIORS_BY_DOMAIN, PRIOR_DIMS, POSTS_BY_DOMAIN

class FakeEnv:

    def __init__(self, true_environment, 
                    static_fns, num_networks=7, 
                    num_elites = 5, hidden_dim = 220, 
                    safe_config=None,
                    session = None):
        
        self.domain = true_environment.domain
        self.env = true_environment
        self.obs_dim = np.prod(self.env.observation_space.shape)
        self.act_dim = np.prod(self.env.action_space.shape)
        self.active_obs_dim = int(self.obs_dim/self.env.stacks)
        self._session = session

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

        self._model = construct_model(obs_dim_in=self.obs_dim, 
                                        obs_dim_out=self.active_obs_dim,
                                        prior_dim=prior_dim,
                                        act_dim=self.act_dim, 
                                        hidden_dim=hidden_dim, 
                                        num_networks=num_networks, 
                                        num_elites=num_elites,
                                        weights=self.target_weights,
                                        session=self._session)
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
        else:
            inputs = np.concatenate((obs, act), axis=-1)

        ensemble_model_means, ensemble_model_vars = self._model.predict(inputs, factored=True)       #### self.model outputs whole ensembles outputs

        ensemble_disagreement_means = np.nanvar(ensemble_model_means, axis=0)*self.target_weights
        ensemble_disagreement_stds = np.sqrt(np.nanvar(ensemble_model_vars, axis=0)*self.target_weights)
        ensemble_disagreement = np.sum(ensemble_disagreement_means+ensemble_disagreement_stds, axis=-1)
        
        ensemble_model_means[:,:,:-1] += obs[:,-unstacked_obs_size:]           #### models output state change rather than state completely
        ensemble_model_stds = np.sqrt(ensemble_model_vars)                                          #### std = sqrt(variance)
        
        ### directly use means, if deterministic
        if deterministic:
            ensemble_samples = ensemble_model_means                     
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        #### choose one model from ensemble randomly
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = self._model.random_inds(batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[model_inds, batch_inds]
        model_means = ensemble_model_means[model_inds, batch_inds]
        model_stds = ensemble_model_stds[model_inds, batch_inds]
        ####

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        #### retrieve r and done for new state
        rewards, next_obs = samples[:,-1:], samples[:,:-1]


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

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev, 'ensemble_disagreement':ensemble_disagreement}
        return next_obs, rewards, terminals, info

    def train(self, samples, **kwargs):
        # check priors
        priors = self.prior_f(samples['observations'], samples['actions']) if self.prior_f else None

        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
        train_inputs, train_outputs = format_samples_for_training(samples, priors=priors, safe_config=self.safe_config, add_noise=True)
        model_metrics = self._model.train(train_inputs, 
                                            train_outputs, 
                                            **kwargs)
        return model_metrics


    def reset_model(self):
        self._model.reset()

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

