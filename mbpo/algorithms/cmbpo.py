## adapted from https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py

import os
import math
import pickle
from collections import OrderedDict
from collections import defaultdict
from numbers import Number
from itertools import count
import gtimer as gt
import pdb
import random
import sys
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util


from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning.replay_pools.mjc_state_replay_pool import MjcStateReplayPool
from softlearning.replay_pools.modelbuffer import ModelBuffer
from softlearning.replay_pools.cpobuffer import CPOBuffer
from softlearning.samplers.model_sampler import ModelSampler
from softlearning.policies.safe_utils.logx import EpochLogger
from softlearning.policies.safe_utils.utils import EPS

from mbpo.models.constructor import construct_model, format_samples_for_dyn, reset_model
from mbpo.models.fake_env import FakeEnv
from mbpo.models.perturbed_env import PerturbedEnv
from mbpo.utils.writer import Writer
from mbpo.utils.visualization import visualize_policy
from mbpo.utils.logging import Progress, update_dict
import mbpo.utils.filesystem as filesystem


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class CMBPO(RLAlgorithm):
    """Model-Based Policy Optimization (MBPO)

    References
    ----------
        Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine. 
        When to Trust Your Model: Model-Based Policy Optimization. 
        arXiv preprint arXiv:1906.08253. 2019.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            mjc_model_environment,
            policy,
            Qs,
            pool,
            static_fns,
            plotter=None,
            tf_summaries=False,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            eval_every_n_steps=5e3,

            deterministic=False,
            model_train_freq=250,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            dyn_model_train_schedule=[20, 100, 1, 5],
            cost_model_train_schedule=[20, 100, 1, 30],
            cares_about_cost = False,
            dyn_m_discount = 1,
            cost_m_discount = 1,                     
            max_uncertainty_rew = None,
            max_uncertainty_c = None,
            rollout_mode = 'schedule',
            rollout_schedule=[20,100,1,1],
            maxroll = 80,
            max_tddyn_err = 1e-5,
            max_tddyn_err_decay = .995,
            min_real_samples_per_epoch = 1000,
            batch_size_policy = 5000,
            hidden_dims=(200, 200, 200, 200),
            max_model_t=None,

            use_mjc_state_model = False,
            model_std_inc = 0.02,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(CMBPO, self).__init__(**kwargs)

        self.obs_space = training_environment.observation_space
        self.act_space = training_environment.action_space
        self.obs_dim = np.prod(training_environment.observation_space.shape)
        
        self.act_dim = np.prod(training_environment.action_space.shape)

        #### determine unstacked obs dim
        self.num_stacks = training_environment.stacks
        self.stacking_axis = training_environment.stacking_axis
        self.active_obs_dim = int(self.obs_dim/self.num_stacks)

        self._dyn_m_discount = dyn_m_discount
        self._cost_m_discount = cost_m_discount
        self.cares_about_cost = cares_about_cost

        ## create fake environment for model
        self.fake_env = FakeEnv(training_environment,
                                    static_fns, num_networks=7, 
                                    num_elites=5, 
                                    hidden_dims=hidden_dims, 
                                    dyn_discount = self._dyn_m_discount,
                                    cost_m_discount = self._cost_m_discount,
                                    cares_about_cost=cares_about_cost,
                                    session = self._session)

        self.use_mjc_state_model = use_mjc_state_model

        self._rollout_schedule = rollout_schedule
        self._max_model_t = max_model_t

        self._model_retain_epochs = model_retain_epochs
        self.eval_every_n_steps= eval_every_n_steps

        self._dyn_model_train_schedule = dyn_model_train_schedule
        self._cost_model_train_schedule = cost_model_train_schedule

        self._dyn_model_train_freq = 1
        self._cost_model_train_freq = 1
        self._rollout_batch_size = int(rollout_batch_size)
        self._max_uncertainty_rew = max_uncertainty_rew
        self._max_uncertainty_c = max_uncertainty_c
        self._deterministic = deterministic
        self._real_ratio = real_ratio

        self._log_dir = os.getcwd()
        self._writer = Writer(self._log_dir)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._mjc_model_environment = mjc_model_environment
        self.perturbed_env = PerturbedEnv(self._mjc_model_environment, std_inc=model_std_inc)
        self._policy = policy
        self._initial_exploration_policy = policy   #overwriting initial _exploration policy, not implemented for cpo yet

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        # set up pool
        pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in self._policy.pi_info_phs.items()}
        self._pool.initialize(pi_info_shapes,
                                gamma = self._policy.gamma,
                                lam = self._policy.lam,
                                cost_gamma = self._policy.cost_gamma,
                                cost_lam = self._policy.cost_lam)

        self._policy_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('[ MBPO ] Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape
        

        ### model sampler and buffer
        self.rollout_mode = rollout_mode
        self.max_tddyn_err = max_tddyn_err
        self.max_tddyn_err_decay = max_tddyn_err_decay
        self.min_real_samples_per_epoch = min_real_samples_per_epoch
        self.batch_size_policy = batch_size_policy

        self.model_pool = ModelBuffer(batch_size=self._rollout_batch_size, 
                                        max_path_length=maxroll, 
                                        env = self.fake_env,
                                        ensemble_size=num_networks,
                                        rollout_mode = self.rollout_mode,
                                        cares_about_cost = cares_about_cost,
                                        max_uncertainty_c = self._max_uncertainty_c,
                                        max_uncertainty_r = self._max_uncertainty_rew,                                        
                                        )
        self.model_pool.initialize(pi_info_shapes,
                                    gamma = self._policy.gamma,
                                    lam = self._policy.lam,
                                    cost_gamma = self._policy.cost_gamma,
                                    cost_lam = self._policy.cost_lam,
                                    ) 
        #@anyboby debug
        self.model_sampler = ModelSampler(max_path_length=maxroll,
                                            batch_size=self._rollout_batch_size,
                                            store_last_n_paths=10,
                                            cares_about_cost = cares_about_cost,
                                            max_uncertainty_c = self._max_uncertainty_c,
                                            max_uncertainty_r = self._max_uncertainty_rew,
                                            logger=None,
                                            rollout_mode = self.rollout_mode,
                                            )

        # provide policy and sampler with the same logger
        self.logger = EpochLogger()
        self._policy.set_logger(self.logger)    
        self.sampler.set_logger(self.logger)
        #self.model_sampler.set_logger(self.logger)

    def _train(self):
        
        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """

        #### pool is e.g. simple_replay_pool
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool

        if not self._training_started:
            #### perform some initial steps (gather samples) using initial policy
            ######  fills pool with _n_initial_exploration_steps samples
            self._initial_exploration_hook(
                training_environment, self._policy, pool)
        
        #### set up sampler with train env and actual policy (may be different from initial exploration policy)
        ######## note: sampler is set up with the pool that may be already filled from initial exploration hook
        self.sampler.initialize(training_environment, policy, pool)
        self.model_sampler.initialize(self.fake_env, policy, self.model_pool)

        #### reset gtimer (for coverage of project development)
        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)
        self.policy_epoch = 0       ### count policy updates
        self.new_real_samples = 0
        self.last_eval_step = 0
        self.diag_counter = 0
        running_diag = {}
        self.approx_model_batch = 0.5*self.batch_size_policy    ### some size to start off

        #### not implemented, could train policy before hook
        self._training_before_hook()

        #### iterate over epochs, gt.timed_for to create loop with gt timestamps
        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            
            #### do something at beginning of epoch (in this case reset self._train_steps_this_epoch=0)
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            #### util class Progress, e.g. for plotting a progress bar
            #######   note: sampler may already contain samples in its pool from initial_exploration_hook or previous epochs
            self._training_progress = Progress(self._epoch_length * self._n_train_repeat/self._train_every_n_steps)


            samples_added = 0
            #=====================================================================#
            #            Rollout model                                            #
            #=====================================================================#
            model_samples = None
            keep_rolling = True
            model_metrics = {}
            #### start model rollout
            if self._real_ratio<1.0: #if self._timestep % self._model_train_freq == 0 and self._real_ratio < 1.0:
                if self.rollout_mode == 'schedule':
                    self._set_rollout_length()

                while keep_rolling:
                    #=====================================================================#
                    #  Rollout Model                                                      #
                    #=====================================================================#
                    print('[ Model Rollout ] Starting | Epoch: {} | Batch size: {}'.format(
                        self._epoch, self._rollout_batch_size 
                    ))                

                    #=====================================================================#
                    #                           Model Rollouts                            #
                    #=====================================================================#
                    # rand_inds = np.random.randint(0, len(real_samples[0]), self._rollout_batch_size)
                    # start_states = real_samples[0][rand_inds]
                    start_states = self._pool.rand_batch_from_archive(self._rollout_batch_size, fields=['observations'])['observations']
                    self.model_sampler.reset(start_states)
                    if self.rollout_mode=='uncertainty':
                        self.model_sampler.set_max_uncertainty(self.max_tddyn_err)

                    for i in count():
                        print(f'Model Sampling step Nr. {i+1}')

                        _,_,_,info = self.model_sampler.sample()
                        alive_ratio = info.get('alive_ratio', 1)

                        ### stop rollout depending on rollout_mode
                        if self.rollout_mode=='schedule':
                            if i+1>self._rollout_length or \
                                    self.model_sampler._total_samples + samples_added >= self.approx_model_batch:
                                break
                        elif self.rollout_mode=='iv_gae':
                            if alive_ratio<0.2:
                                print(f'Stopping Rollout at step {i+1}')
                                break
                        else:
                            if alive_ratio<0.2 or \
                                    self.model_sampler._total_samples + samples_added >= self.approx_model_batch:
                                print(f'Stopping Rollout at step {i+1}')
                                break
                    
                    ### diagnostics for rollout ###
                    rollout_diagnostics = self.model_sampler.finish_all_paths()

                    ######################################################################
                    ### get model_samples, get() invokes the inverse variance rollouts ###
                    model_samples_new, buffer_diagnostics_new = self.model_pool.get()
                    model_samples = [np.concatenate((o,n), axis=0) for o,n in zip(model_samples, model_samples_new)] if model_samples else model_samples_new

                    #model_metrics.update(buffer_diagnostics)
                    new_n_samples = len(model_samples_new[0])
                    model_metrics = update_dict(model_metrics, rollout_diagnostics, weight_a= samples_added/(new_n_samples+samples_added),weight_b=new_n_samples/(new_n_samples+samples_added))

                    ######################################################################

                    ### run diagnostics on model data
                    if buffer_diagnostics_new['poolm_batch_size']>0:
                        model_data_diag = self._policy.run_diagnostics(model_samples_new)
                        model_data_diag = {k+'_m':v for k,v in model_data_diag.items()}
                        #model_metrics.update(model_data_diag)
                    model_metrics = update_dict(model_metrics, model_data_diag, weight_a= samples_added/(new_n_samples+samples_added),weight_b=new_n_samples/(new_n_samples+samples_added))
                    model_metrics = update_dict(model_metrics, buffer_diagnostics_new, weight_a= samples_added/(new_n_samples+samples_added),weight_b=new_n_samples/(new_n_samples+samples_added))
                    
                    samples_added += new_n_samples
                    model_metrics.update({'samples_added':samples_added})

                    td_dyn_err = model_metrics.get('poolm_td_dyn_n', EPS)
            
                    self.approx_model_batch = min(self.max_tddyn_err*self.batch_size_policy / td_dyn_err, self.batch_size_policy-self.min_real_samples_per_epoch)
                    keep_rolling = not(samples_added > .95*self.approx_model_batch)
                    
                gt.stamp('epoch_rollout_model')
                model_metrics.update({'max_tddyn_err':self.max_tddyn_err})

            #=====================================================================#
            #  Sample                                                             #
            #=====================================================================#
            
            n_real_samples = max(self.batch_size_policy-samples_added, self.min_real_samples_per_epoch)
            model_metrics.update({'n_real_samples':n_real_samples})
            start_samples = self.sampler._total_samples                     
            ### train for epoch_length ###
            for i in count():

                #### _timestep is within an epoch
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                #### not implemented atm
                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                ##### śampling from the real world ! #####
                _,_, _, _ = self._do_sampling(timestep=self.policy_epoch)
                gt.stamp('sample')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

                if self.ready_to_train or self._timestep>n_real_samples:
                    self.sampler.finish_all_paths(append_val=True)
                    self.new_real_samples += self._timestep
                    break

            #=====================================================================#
            #  Train model                                                        #
            #=====================================================================#
            if self.new_real_samples>1024 and self._real_ratio<1.0:
                model_diag = self.train_model(min_epochs=15, max_epochs=150)
                self.new_real_samples = 0
                model_metrics.update(model_diag)

            #=====================================================================#
            #  Get Buffer Data                                                    #
            #=====================================================================#
            real_samples, buf_diag = self._pool.get()

            ### run diagnostics on real data
            policy_diag = self._policy.run_diagnostics(real_samples)
            policy_diag = {k+'_r':v for k,v in policy_diag.items()}
            model_metrics.update(policy_diag)
            model_metrics.update(buf_diag)


            #=====================================================================#
            #  Update Policy                                                      #
            #=====================================================================#
            train_samples = [np.concatenate((r,m), axis=0) for r,m in zip(real_samples, model_samples)] if model_samples else real_samples

            self._policy.update_policy(train_samples)
            self._policy.update_critic(train_samples)
            
            self.policy_epoch += 1
            self.max_tddyn_err *= self.max_tddyn_err_decay
            #### log policy diagnostics
            self._policy.log()

            gt.stamp('train')
            #=====================================================================#
            #  Log performance and stats                                          #
            #=====================================================================#

            self.sampler.log()
            # self.logger.log_tabular('Epoch', self._epoch)
            # write results to file, ray prints for us, so no need to print from logger
            logger_diagnostics = self.logger.dump_tabular(output_dir=self._log_dir, print_out=False)

            #=====================================================================#

            if self._total_timestep // self.eval_every_n_steps > self.last_eval_step:
                evaluation_paths = self._evaluation_paths(
                    policy, evaluation_environment)
                gt.stamp('evaluation_paths')
                
                self.last_eval_step = self._total_timestep // self.eval_every_n_steps
            else: 
                evaluation_paths = []

            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
                diag_obs_batch = np.concatenate(([evaluation_paths[i]['observations'] for i in range(len(evaluation_paths))]), axis=0)
            else:
                evaluation_metrics = {}
                diag_obs_batch = []

            gt.stamp('epoch_after_hook')

            # sampler_diagnostics = self.sampler.get_diagnostics()

            new_diagnostics = {}

            time_diagnostics = gt.get_times().stamps.itrs  

            # add diagnostics from logger
            new_diagnostics.update(logger_diagnostics) 

            new_diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'model/{key}', model_metrics[key])
                    for key in sorted(model_metrics.keys())
                ),
            )))

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)

            #### updateing and averaging
            old_ts_diag = running_diag.get('timestep', 0)
            new_ts_diag = self._total_timestep-self.diag_counter-old_ts_diag
            w_olddiag = old_ts_diag/(new_ts_diag+old_ts_diag)
            w_newdiag = new_ts_diag/(new_ts_diag+old_ts_diag)
            running_diag = update_dict(running_diag, new_diagnostics, weight_a=w_olddiag, weight_b=w_newdiag)
            running_diag.update({'timestep':new_ts_diag + old_ts_diag})
            ####
            
            if new_ts_diag + old_ts_diag > self.eval_every_n_steps:
                running_diag.update({
                    'epoch':self._epoch,
                    'timesteps_total':self._total_timestep,
                    'train-steps':self._num_train_steps,
                })
                self.diag_counter = self._total_timestep
                diag = running_diag.copy() 
                running_diag = {}
                yield diag

        self.sampler.terminate()

        self._training_after_hook()

        self._training_progress.close()

        ### this is where we yield the episode diagnostics to tune trial runner ###
        yield {'done': True, **self.running_diag}

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _initial_exploration_hook(self, env, initial_exploration_policy, pool):
        if self._n_initial_exploration_steps < 1: return

        if not initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        self.sampler.initialize(env, initial_exploration_policy, pool)
        while True:
            self.sampler.sample(timestep=0)
            if self.sampler._total_samples >= self._n_initial_exploration_steps:
                self.sampler.finish_all_paths(append_val=True)
                pool.get()  # moves policy samples to archive
                break
        
        ### train model
        if self._real_ratio<1.0:
            self.train_model(min_epochs=150, max_epochs=500)

        ### train critic
        critic_samples = self._pool.get_archive([     #### include initial expl. hook samples
                'observations',
                'actions',
                'advantages',
                'return_vars',
                'cadvantages',
                'creturn_vars',
                'returns',
                'creturns',
                'log_policies',
                'values',
                'value_vars',
                'cvalues',
                'cvalue_vars',
                'costs',
                'pi_infos',
            ])
        critic_samples = list(critic_samples.values())       ### samples from initial exploration hook
        
        self._policy.update_critic(critic_samples, 
                                    min_epoch_before_break = 10, 
                                    max_epochs=350)

    def train_model(self, min_epochs=50, max_epochs=500, batch_size=2048):
        self._dyn_model_train_freq = self._set_model_train_freq(
            self._dyn_model_train_freq, 
            self._dyn_model_train_schedule
            ) ## set current train freq.
        self._cost_model_train_freq = self._set_model_train_freq(
            self._cost_model_train_freq, 
            self._cost_model_train_schedule
            )

        print('[ MBPO ] log_dir: {} | ratio: {}'.format(self._log_dir, self._real_ratio))
        print('[ MBPO ] Training model at epoch {} | freq {} | timestep {} (total: {}) (total: {})'.format(
            self._epoch, self._dyn_model_train_freq, self._timestep, self._total_timestep, self._num_train_steps)
        )

        model_samples = self._pool.get_archive(['observations',
                                                'actions',
                                                'next_observations',
                                                'rewards',
                                                'costs',
                                                'terminals',
                                                'epochs',
                                                ])

        if self._epoch%self._dyn_model_train_freq==0:
            diag_dyn = self.fake_env.train_dyn_model(
                model_samples, 
                discount = self._dyn_m_discount,
                batch_size=batch_size, #512
                max_epochs=max_epochs, # max_epochs 
                min_epoch_before_break=min_epochs, # min_epochs, 
                holdout_ratio=0.2, 
                max_t=self._max_model_t
                )

        if self._epoch%self._cost_model_train_freq==0 and self.fake_env.cares_about_cost:
            diag_c = self.fake_env.train_cost_model(
                model_samples, 
                discount = self._cost_m_discount,
                batch_size= batch_size, #batch_size, #512, 
                min_epoch_before_break= min_epochs,#min_epochs,
                max_epochs=max_epochs, # max_epochs, 
                holdout_ratio=0.2, 
                max_t=self._max_model_t
                )
            diag_dyn.update(diag_c)

        return diag_dyn

                
    @property
    def _total_timestep(self):
        total_timestep = self.sampler._total_samples
        return total_timestep

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

   
    def _do_sampling(self, timestep):
        return self.sampler.sample(timestep = timestep)
    
    def _set_model_train_freq(self, var, schedule):
        min_epoch, max_epoch, min_freq, max_freq = schedule
        if self._epoch <= min_epoch:
            y = min_freq
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_freq - min_freq) + min_freq

        var = int(y)
        print('[ Model Train Frequency ] Epoch: {} (min: {}, max: {}) | Frequency: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, var, min_freq, max_freq
        ))
        return var

    def _visualize_model(self, env, timestep):
        ## save env state
        state = env.unwrapped.state_vector()
        qpos_dim = len(env.unwrapped.sim.data.qpos)
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]

        print('[ Visualization ] Starting | Epoch {} | Log dir: {}\n'.format(self._epoch, self._log_dir))
        visualize_policy(env, self.fake_env, self._policy, self._writer, timestep)
        print('[ Visualization ] Done')
        ## set env state
        env.unwrapped.set_state(qpos, qvel)


    def get_diagnostics(self,
                        iteration,
                        obs_batch = None,
                        training_paths = None,
                        evaluation_paths = None):
        """Return diagnostic information as ordered dictionary.

        Records state value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        # @anyboby 
        warnings.warn('diagnostics not implemented yet!')

        # diagnostics = OrderedDict({
        #     })

        # policy_diagnostics = self._policy.get_diagnostics(                  #use eval paths
        #     obs_batch[:,-self.active_obs_dim:])
        # diagnostics.update({
        #     f'policy/{key}': value
        #     for key, value in policy_diagnostics.items()
        # })

        # if self._plotter:
        #     self._plotter.draw()

        diagnostics = {}

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            self._policy.tf_saveables
        }

        # saveables = {
        #     '_policy_optimizer': self._policy_optimizer,
        #     **{
        #         f'Q_optimizer_{i}': optimizer
        #         for i, optimizer in enumerate(self._Q_optimizers)
        #     },
        #     '_log_alpha': self._log_alpha,
        # }

        # if hasattr(self, '_alpha_optimizer'):
        #     saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
