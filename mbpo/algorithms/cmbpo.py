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

from mbpo.models.constructor import construct_model, format_samples_for_dyn, reset_model
from mbpo.models.fake_env import FakeEnv
from mbpo.models.perturbed_env import PerturbedEnv
from mbpo.utils.writer import Writer
from mbpo.utils.visualization import visualize_policy
from mbpo.utils.logging import Progress
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

            deterministic=False,
            model_train_freq=250,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            rollout_schedule=[20,100,1,1],
            dyn_model_train_schedule=[20, 100, 1, 5],
            cost_model_train_schedule=[20, 100, 1, 30],
            dyn_m_discount = 1,
            cost_m_discount = 1,                     
            max_uncertainty_rew = None,
            max_uncertainty_c = None,
            iv_gae = False,
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
        
        ## create fake environment for model
        self.fake_env = FakeEnv(training_environment,
                                    static_fns, num_networks=7, 
                                    num_elites=5, 
                                    hidden_dims=hidden_dims, 
                                    dyn_discount = self._dyn_m_discount,
                                    cost_m_discount = self._cost_m_discount,
                                    cares_about_cost=True,
                                    session = self._session)

        self.use_mjc_state_model = use_mjc_state_model

        self._rollout_schedule = rollout_schedule
        self._max_model_t = max_model_t

        self._model_retain_epochs = model_retain_epochs

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
        self.iv_gae = iv_gae
        self.model_pool = ModelBuffer(batch_size=self._rollout_batch_size, 
                                        max_path_length=60, 
                                        env = self.fake_env,
                                        ensemble_size=num_networks,
                                        iv_gae = self.iv_gae,
                                        )
        self.model_pool.initialize(pi_info_shapes,
                                    gamma = self._policy.gamma,
                                    lam = self._policy.lam,
                                    cost_gamma = self._policy.cost_gamma,
                                    cost_lam = self._policy.cost_lam,
                                    ) 
        #@anyboby debug
        self.model_sampler = ModelSampler(max_path_length=60,
                                            batch_size=self._rollout_batch_size,
                                            store_last_n_paths=10,
                                            preprocess_type='default',
                                            max_uncertainty_c = self._max_uncertainty_c,
                                            max_uncertainty_r = self._max_uncertainty_rew,
                                            logger=None,
                                            iv_gae =self.iv_gae,
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
        model_metrics = {}

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

        #### not implemented, could train policy before hook
        self._training_before_hook()
        train_samples = None 

        #### iterate over epochs, gt.timed_for to create loop with gt timestamps
        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            
            #### do something at beginning of epoch (in this case reset self._train_steps_this_epoch=0)
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            #### util class Progress, e.g. for plotting a progress bar
            #######   note: sampler may already contain samples in its pool from initial_exploration_hook or previous epochs
            self._training_progress = Progress(self._epoch_length * self._n_train_repeat/self._train_every_n_steps)

            min_samples = 20e3
            max_samples = 180e3
            samples_added = 0

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

                
                if self.ready_to_train:
                    self.sampler.finish_all_paths(append_val=True)
                    break

            #=====================================================================#
            #  Get Buffer Data                                                    #
            #=====================================================================#
            #### get samples from buffer
            if self._epoch==0:
                arch_samples = self._pool.get_archive([     #### include initial expl. hook samples
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
                arch_samples = list(arch_samples.values())       ### samples from initial exploration hook
                real_samples, buf_diag = self._pool.get()        ### samples from first epoch
                real_samples = [np.concatenate((r,m), axis=0) for r,m in zip(real_samples, arch_samples)]       ### merge
                
                #### little trick: circumvent Trust Region critic update in first run to introduce some starting uncertainty
                self._policy.update_critic(real_samples, 
                                            kl_cliprange=1e5, 
                                            min_epoch_before_break = 10, 
                                            max_epochs=100)

            else:
                real_samples, buf_diag= self._pool.get()

            ### run diagnostics on real data
            policy_diag = self._policy.run_diagnostics(real_samples)
            policy_diag = {k+'_r':v for k,v in policy_diag.items()}
            model_metrics.update(policy_diag)
            model_metrics.update(buf_diag)

            #=====================================================================#
            #  Train and Rollout model                                            #
            #=====================================================================#
            model_samples = None
            
            #### start model rollout
            if self._real_ratio<1.0: #if self._timestep % self._model_train_freq == 0 and self._real_ratio < 1.0:
                self._training_progress.pause()
                self._dyn_model_train_freq = self._set_model_train_freq(
                    self._dyn_model_train_freq, 
                    self._dyn_model_train_schedule
                    ) ## set current train freq.
                self._cost_model_train_freq = self._set_model_train_freq(
                    self._cost_model_train_freq, 
                    self._cost_model_train_schedule
                    )
                print('[ MBPO ] log_dir: {} | ratio: {}'.format(self._log_dir, self._real_ratio))
                print('[ MBPO ] Training model at epoch {} | freq {} | timestep {} (total: {}) | epoch train steps: {} (total: {})'.format(
                    self._epoch, self._dyn_model_train_freq, self._timestep, self._total_timestep, self._train_steps_this_epoch, self._num_train_steps)
                )

                samples = self._pool.get_archive(['observations',
                                                        'actions',
                                                        'next_observations',
                                                        'rewards',
                                                        'costs',
                                                        'terminals',
                                                        'epochs',
                                                        ])
                # if len(samples['observations'])>25000:
                #     dyn_samples = {k:v[-25000:] for k,v in samples.items()} 
                # if len(samples['observations'])>10000:
                #     cost_samples = {k:v[-10000:] for k,v in samples.items()} 
    
                #self.fake_env.reset_model()    # this behaves weirdly
                min_epochs = 150 if self._epoch<10 else 5        ### overtrain a little in the beginning to jumpstart uncertainty prediction
                max_epochs = 500 if self._epoch<10 else 10
                # # if len(samples['observations'])>30000:
                # #     samples = {k:v[-30000:] for k,v in samples.items()} 
                # batch_size = 512 + min(self._epoch//50*512, 7*512)
                batch_size = 2048

                if self._epoch%self._dyn_model_train_freq==0:
                    model_train_metrics_dyn = self.fake_env.train_dyn_model(
                        samples, 
                        discount = self._dyn_m_discount,
                        batch_size=batch_size, #512
                        max_epochs=max_epochs, # max_epochs 
                        min_epoch_before_break=min_epochs, # min_epochs, 
                        holdout_ratio=0.2, 
                        max_t=self._max_model_t
                        )
                    model_metrics.update(model_train_metrics_dyn)

                if self._epoch%self._cost_model_train_freq==0 and self.fake_env.cares_about_cost:
                    model_train_metrics_cost = self.fake_env.train_cost_model(
                        samples, 
                        discount = self._cost_m_discount,
                        batch_size= batch_size, #batch_size, #512, 
                        min_epoch_before_break= min_epochs,#min_epochs,
                        max_epochs=max_epochs, # max_epochs, 
                        holdout_ratio=0.2, 
                        max_t=self._max_model_t
                        )
                    model_metrics.update(model_train_metrics_cost)

                gt.stamp('epoch_train_model')

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
                
                for i in count():
                    print(f'Model Sampling step Nr. {i+1}')

                    _,_,_,info = self.model_sampler.sample()
                    alive_ratio = info.get('alive_ratio', 1)

                    if alive_ratio<0.2 or \
                        self.model_sampler._total_samples + samples_added >= max_samples-alive_ratio*self._rollout_batch_size:                         
                        print(f'Stopping Rollout at step {i+1}')
                        break
                
                ### diagnostics for rollout ###
                rollout_diagnostics = self.model_sampler.finish_all_paths()
                                    
                
                model_metrics.update(rollout_diagnostics)

                ######################################################################
                ### get model_samples, get() invokes the inverse variance rollouts ###
                model_samples, buffer_diagnostics = self.model_pool.get()
                model_metrics.update(buffer_diagnostics)
                samples_added += buffer_diagnostics['poolm_batch_size']
                ######################################################################

                ### run diagnostics on model data
                if buffer_diagnostics['poolm_batch_size']>0:
                    model_data_diag = self._policy.run_diagnostics(model_samples)
                    model_data_diag = {k+'_m':v for k,v in model_data_diag.items()}
                    model_metrics.update(model_data_diag)

                gt.stamp('epoch_rollout_model')
                
                #=====================================================================#
                #                           Model accuracy measurement                #
                #=====================================================================#
                # rand_inds = np.random.randint(0, len(real_samples[0]), self._rollout_batch_size)
                # start_states = real_samples[0][rand_inds]
                start_states = real_samples[0][0::real_samples[0].shape[0]//100+1]

                self.model_sampler.reset(start_states)
                
                for i in count():
                    # print(f'Model Sampling step Nr. {i+1}')

                    _,_,_,info = self.model_sampler.sample()
                    alive_ratio = info.get('alive_ratio', 1)

                    if alive_ratio<0.2 or \
                        self.model_sampler._total_samples + samples_added >= max_samples-alive_ratio*self._rollout_batch_size:                         
                        print(f'Stopping Measurement Rollout at step {i+1}')
                        break
                
                ### diagnostics for rollout ###
                rollout_diagnostics = self.model_sampler.finish_all_paths()
                                    
                
                # model_metrics.update(rollout_diagnostics)

                ######################################################################
                ### get model_samples, get() invokes the inverse variance rollouts ###
                measure_samples, measure_diagnostics = self.model_pool.get()
                measure_diagnostics = {'measure/'+k:v for k,v in measure_diagnostics.items()}
                model_metrics.update(measure_diagnostics)
                samples_added += measure_diagnostics['measure/poolm_batch_size']
                ######################################################################

                ### norm adv var ratio
                adv_var_m = measure_diagnostics['measure/poolm_norm_adv_var']
                cadv_var_m = measure_diagnostics['measure/poolm_norm_cadv_var']
                adv_var_r = buf_diag['poolr_norm_adv_var']
                cadv_var_r = buf_diag['poolr_norm_cadv_var']

                adv_var_ratio = adv_var_m / adv_var_r
                cadv_var_ratio = cadv_var_m / cadv_var_r
                
                gt.stamp('epoch_measure_rollouts')

                ### run diagnostics on model data
                if measure_diagnostics['measure/poolm_batch_size']>0:
                    measure_data_diag = self._policy.run_diagnostics(measure_samples)
                    measure_data_diag = {'measure/'+k:v for k,v in measure_data_diag.items()}
                    measure_data_diag.update({
                        'measure/norm_adv_var_ratio':adv_var_ratio,
                        'measure/norm_cadv_var_ratio':cadv_var_ratio,
                    })

                    model_metrics.update(measure_data_diag)


            if train_samples is None:
                train_samples = [np.concatenate((r,m), axis=0) for r,m in zip(real_samples, model_samples)] if model_samples else real_samples
            else: 
                new_samples = [np.concatenate((r,m), axis=0) for r,m in zip(real_samples, model_samples)] if model_samples else real_samples
                train_samples = [np.concatenate((t,n), axis=0) for t,n in zip(train_samples, new_samples)]

            self._training_progress.resume()

            #=====================================================================#
            #  Update Policy                                                      #
            #=====================================================================#
            if (len(train_samples[0])>=min_samples or self._epoch==0):     ### @anyboby TODO kickstarting at the beginning for logger (change this !)
                self._policy.update_policy(train_samples)
                self._policy.update_critic(train_samples)
                
                self.policy_epoch += 1

                #### empty train_samples
                train_samples = None
                samples_added = 0

                #### log policy diagnostics
                self._policy.log()
                model_metrics.update({'Policy Update?':1})
            else: 
                model_metrics.update({'Policy Update?':0})

            gt.stamp('train')
            #=====================================================================#
            #  Log performance and stats                                          #
            #=====================================================================#

            self.sampler.log()
            self.logger.log_tabular('Epoch', self._epoch)
            # write results to file, ray prints for us, so no need to print from logger
            logger_diagnostics = self.logger.dump_tabular(output_dir=self._log_dir, print_out=False)

            #=====================================================================#

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            gt.stamp('training_paths')
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment)
            gt.stamp('evaluation_paths')

            training_metrics = self._evaluate_rollouts(
                training_paths, training_environment)
            gt.stamp('training_metrics')
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}


            self._epoch_after_hook(training_paths)
            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            diag_obs_batch = np.concatenate(([evaluation_paths[i]['observations'] for i in range(len(evaluation_paths))]), axis=0)
            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                obs_batch=diag_obs_batch,
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            # add diagnostics from logger
            diagnostics.update(logger_diagnostics)

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                *(
                    (f'model/{key}', model_metrics[key])
                    for key in sorted(model_metrics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)

            yield diagnostics

        self.sampler.terminate()

        self._training_after_hook()

        self._training_progress.close()

        ### this is where we yield the episode diagnostics to tune trial runner ###
        yield {'done': True, **diagnostics}

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _initial_exploration_hook(self, env, initial_exploration_policy, pool):
        if self._n_initial_exploration_steps < 1: return

        if not initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        self.sampler.initialize(env, initial_exploration_policy, pool)
        while pool.arch_size < self._n_initial_exploration_steps:
            self.sampler.sample(timestep=0)
            if self.ready_to_train:
                self.sampler.finish_all_paths(append_val=True)
                pool.get()  # moves policy samples to archive
                
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
                        obs_batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records state value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        # @anyboby 
        warnings.warn('diagnostics not implemented yet!')

        diagnostics = OrderedDict({
            })

        policy_diagnostics = self._policy.get_diagnostics(                  #use eval paths
            obs_batch[:,-self.active_obs_dim:])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

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
