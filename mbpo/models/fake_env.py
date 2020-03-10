import numpy as np
import tensorflow as tf
import pdb

class FakeEnv:

    def __init__(self, model, static_fns, safe_config=None):
        self.model = model
        self.static_fns = static_fns
        self.safe_config = safe_config
        if safe_config:
            self.stacks = self.safe_config['stacks']
            self.stacking_axis = self.safe_config['stacking_axis']
        else:
            self.stacks = 1
            self.stacking_axis = 0
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
            return_single = True
        else:
            return_single = False


        unstacked_obs_size = int(obs.shape[1+self.stacking_axis]/self.stacks)               ### e.g. if a stacked obs is 88 with 4 stacks,
                                                                                            ### unstacking it yields 22

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)       #### self.model outputs whole ensembles outputs
        test = obs[:,-unstacked_obs_size:]
        ensemble_model_means[:,:,:-1] += obs[:,-unstacked_obs_size:]                                #### models output state change rather than state completely
        ensemble_model_stds = np.sqrt(ensemble_model_vars)                                          #### std = sqrt(variance)

        ### directly use means, if deterministic
        if deterministic:
            ensemble_samples = ensemble_model_means                     
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        #### choose one model from ensemble randomly
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = self.model.random_inds(batch_size)
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

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,-1:], terminals, model_means[:,:-1]), axis=-1)
        return_stds = np.concatenate((model_stds[:,-1:], np.zeros((batch_size,1)), model_stds[:,:-1]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

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



