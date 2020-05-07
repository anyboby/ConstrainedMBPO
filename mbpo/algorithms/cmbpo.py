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

from mbpo.models.constructor import construct_model, format_samples_for_training, reset_model
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
            max_uncertainty = None,
            hidden_dim=200,
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
        self.safe_config = training_environment.safeconfig if hasattr(training_environment, 'safeconfig') else None
        if self.safe_config: weighted=True 
        else: weighted=False
        #unstacked_obs_dim[self.stacking_axis] = int(obs_dim[self.stacking_axis]/self.num_stacks)

        # #### create fake env from model 
        # self._model = construct_model(obs_dim_in=self.obs_dim, 
        #                                 obs_dim_out=self.active_obs_dim,
        #                                 act_dim=self.act_dim, 
        #                                 hidden_dim=hidden_dim, 
        #                                 num_networks=num_networks, 
        #                                 num_elites=num_elites,
        #                                 weighted=True)
        # self._static_fns = static_fns           # termination functions for the envs (model can't simulate those)
        # self.fake_env = FakeEnv(self._model, self._static_fns, safe_config=self.safe_config)
        
        ## create fake environment for model
        self.fake_env = FakeEnv(training_environment,
                                    static_fns, num_networks=7, 
                                    num_elites=5, hidden_dim=hidden_dim, 
                                    cares_about_cost=True,
                                    safe_config=self.safe_config,
                                    session = self._session)

        self.use_mjc_state_model = use_mjc_state_model

        self._rollout_schedule = rollout_schedule
        self._max_model_t = max_model_t

        # self._model_pool_size = model_pool_size
        # print('[ MBPO ] Model pool size: {:.2E}'.format(self._model_pool_size))
        # self._model_pool = SimpleReplayPool(pool._observation_space, pool._action_space, self._model_pool_size)

        self._model_retain_epochs = model_retain_epochs

        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._max_uncertainty = max_uncertainty
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

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

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
        self._Q_lr = lr

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
        self.model_pool = ModelBuffer(batch_size=self._rollout_batch_size, 
                                        max_path_length=40, 
                                        observation_space = self.obs_space, 
                                        action_space = self.act_space)
        self.model_pool.initialize(pi_info_shapes,
                                gamma = self._policy.gamma,
                                lam = self._policy.lam,
                                cost_gamma = self._policy.cost_gamma,
                                cost_lam = self._policy.cost_lam)        
        
        # self.model_pool_debug = CPOBuffer(size=1000, 
        #                 archive_size=100000, 
        #                 observation_space = self.obs_space, 
        #                 action_space = self.act_space)
        
        # self.model_pool_debug.initialize(pi_info_shapes,
        #                 gamma = self._policy.gamma,
        #                 lam = self._policy.lam,
        #                 cost_gamma = self._policy.cost_gamma,
        #                 cost_lam = self._policy.cost_lam)     

        
        self.model_sampler = ModelSampler(max_path_length=40,
                                            batch_size=self._rollout_batch_size,
                                            store_last_n_paths=10,
                                            preprocess_type='default',
                                            max_uncertainty = self._max_uncertainty,
                                            logger=None,
                                            )

        # provide session to policy and agent
        self._policy.prepare_session(self._session)

        # provide policy and sampler with the same logger
        self.logger = EpochLogger()
        self._policy.set_logger(self.logger)
        self.sampler.set_logger(self.logger)
        #self.model_sampler.set_logger(self.logger)


    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()

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

        #### init Qs for SAC
        if not self._training_started:
            self._init_training()

            #### perform some initial steps (gather samples) using initial policy
            ######  fills pool with _n_initial_exploration_steps samples
            self._initial_exploration_hook(
                training_environment, self._policy, pool)
            self.sampler.finish_all_paths(append_val=True)
            pool.dump_to_archive() # move old policy samples to archive

        
        
        #### set up sampler with train env and actual policy (may be different from initial exploration policy)
        ######## note: sampler is set up with the pool that may be already filled from initial exploration hook
        self.sampler.initialize(training_environment, policy, pool)
        self.model_sampler.initialize(self.fake_env, policy, self.model_pool)
        # self.model_sampler.set_debug_buf(self.model_pool_debug)

        #### reset gtimer (for coverage of project development)
        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

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

            min_samples = 50e3
            max_samples = 220e3
                
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
                _,_, _, _ = self._do_sampling(timestep=self._total_timestep)
                gt.stamp('sample')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

                
                if self.ready_to_train:
                    self.sampler.finish_all_paths(append_val=True)
                    break

            #=====================================================================#
            #  Train and Rollout model                                            #
            #=====================================================================#
            model_samples = None
            #### start model rollout
            if self._real_ratio<1.0: #if self._timestep % self._model_train_freq == 0 and self._real_ratio < 1.0:
                self._training_progress.pause()
                print('[ MBPO ] log_dir: {} | ratio: {}'.format(self._log_dir, self._real_ratio))
                print('[ MBPO ] Training model at epoch {} | freq {} | timestep {} (total: {}) | epoch train steps: {} (total: {})'.format(
                    self._epoch, self._model_train_freq, self._timestep, self._total_timestep, self._train_steps_this_epoch, self._num_train_steps)
                )

                samples = self._pool.get_archive(['observations',
                                                        'actions',
                                                        'next_observations',
                                                        'rewards',
                                                        'costs',
                                                        'terminals'])
                #self.fake_env.reset_model()    # this behaves weirdly
                model_train_metrics = self.fake_env.train(samples, batch_size=1024, max_epochs=1500, holdout_ratio=0.2,min_epoch_before_break = 150, max_t=self._max_model_t)
                model_metrics.update(model_train_metrics)
                gt.stamp('epoch_train_model')

                #=====================================================================#
                #  Rollout Model                                                      #
                #=====================================================================#
                self._set_rollout_length()
                print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {}'.format(
                    self._epoch, 'auto', self._rollout_batch_size #self._epoch, self._rollout_length, self._rollout_batch_size
                ))                
                
                ### set initial states
                start_states = self._pool.rand_batch_from_archive(self._rollout_batch_size, fields=['observations'])['observations']
                self.model_sampler.reset(start_states)
                
                for i in count():
                    print(f'Sampling step Nr. {i+1}')

                    _,_,_,info = self.model_sampler.sample()
                    alive_ratio = info.get('alive_ratio', 1)

                    if alive_ratio<0.2 or \
                        self.model_sampler._total_samples >= max_samples-alive_ratio*self._rollout_batch_size: 
                        #samples a bit more than it should
                        
                        print(f'Stopping Rollout at step {i+1}')
                        break
                
                ### diagnostics for rollout ###
                gt.stamp('epoch_rollout_model')
                rollout_diagnostics = self.model_sampler.finish_all_paths()
                model_metrics.update(rollout_diagnostics)

                ### get model_samples after rollout
                model_samples = self.model_pool.get()
            
            real_samples= self._pool.get()

            if train_samples is None:
                train_samples = [np.concatenate((r,m), axis=0) for r,m in zip(real_samples, model_samples)] if model_samples else real_samples
            else: 
                new_samples = [np.concatenate((r,m), axis=0) for r,m in zip(real_samples, model_samples)] if model_samples else real_samples
                train_samples = [np.concatenate((t,n), axis=0) for t,n in zip(train_samples, new_samples)]

            self._training_progress.resume()

            #=====================================================================#
            #  Update Policy                                                      #
            #=====================================================================#
            if len(train_samples[0])>=min_samples or self._epoch<5:     ### @anyboby TODO kickstarting at the beginning for logger (change this !)
                self._policy.update(train_samples)
                gt.stamp('train')

                #### empty train_samples
                train_samples = None

                #### log policy diagnostics
                self._policy.log()
                model_metrics.update({'Policy Update?':1})
            else: 
                model_metrics.update({'Policy Update?':0})

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

    def _log_policy(self):
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        weights = self._policy.get_weights()
        data = {'policy_weights': weights}
        full_path = os.path.join(save_path, 'policy_{}.pkl'.format(self._total_timestep))
        print('Saving policy to: {}'.format(full_path))
        pickle.dump(data, open(full_path, 'wb'))

    def _log_model(self):
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        print('Saving model to: {}'.format(save_path))
        self._model.save(save_path, self._total_timestep)

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

    def _reallocate_model_pool(self, use_mjc_model_pool=False):
        obs_space = self._obs_space
        act_space = self._act_space

        #### calc rollouts per epoch and total steps made in the model per epoch
        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ MBPO ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            if use_mjc_model_pool:
                self._model_pool = MjcStateReplayPool(obs_space, act_space, new_pool_size)
            else: 
                self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)

        elif self._model_pool._max_size != new_pool_size:
            print('[ MBPO ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            if use_mjc_model_pool:
                new_pool = MjcStateReplayPool(obs_space, act_space, new_pool_size)
            else:
                new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _train_model(self, **kwargs):

        env_samples = self._pool.get_archive()

        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
        train_inputs, train_outputs = format_samples_for_training(env_samples, safe_config=self.safe_config, add_noise=True)
        model_metrics = self._model.train(train_inputs, train_outputs, **kwargs)
        return model_metrics

    def _rollout_model(self, rollout_batch_size, **kwargs):
        
        assert self.model_pool.has_room
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size
        ))

        batch = self.sampler.random_batch(rollout_batch_size)
        obs = batch['observations']
        steps_added = []

        if self.use_mjc_state_model:
            cum_error = 0
            sim_states = batch['sim_states']
            ### start rollouts from collected samples 
            for i in range(rollout_batch_size):
                if (i%int(rollout_batch_size/50)==0):
                    print(f'Rollout Progress: {i/rollout_batch_size*100} %', end='', flush=True)

                cur_obs_real = obs[i]
                cur_sim_state = sim_states[i,0]
                cur_obs = self.perturbed_env.reset(sim_state=cur_sim_state)
                new_sim_state = self.perturbed_env.get_sim_state()
                start_errorcheck = np.sum(abs(cur_obs_real-cur_obs))
                cum_error += start_errorcheck
                current_path = defaultdict(list)

                for rollout in range(self._rollout_length):
                    ### get action
                    action = self._policy.actions_np([
                        self._mjc_model_environment.convert_to_active_observation(
                        cur_obs)[None]
                    ])[0]

                    next_obs, rew, term, info = self.perturbed_env.step(action)
                    next_sim_state = self.perturbed_env.get_sim_state

                    processed_sample = {
                        'observations': cur_obs,
                        'actions': action,
                        'rewards': [rew],
                        'terminals': [term],
                        'next_observations': next_obs,
                        'infos': info,
                        'sim_states': [cur_sim_state],
                    }

                    for key, value in processed_sample.items():
                        current_path[key].append(value)
                    
                    steps_added.append(1)
                    if term or rollout==(self._rollout_length-1):
                        ### store samples
                        last_path = {
                            field_name: np.array(values)
                            for field_name, values in current_path.items()
                        }                                    
                        self._model_pool.add_samples(last_path)

                        current_path = defaultdict(list)
                        cur_obs = None
                        cur_sim_state = None
                        break
                    else:
                        cur_obs = next_obs
                        cur_sim_state = next_sim_state
            
            print(f'\n Model rollout with perturbed mjc reset env, acc error: {cum_error}')



        else:    
            log_a = []
            log_obs = []
            log_a.append(batch['actions'])
            log_obs.append(obs)
            for i in range(self._rollout_length):
                act = self._policy.actions_np(obs[:,-self.active_obs_dim:])
                ##### here we're dreaming in the agents model #####
                next_obs, rew, term, info = self.fake_env.step(obs, act, **kwargs)  
                log_a.append(act)
                log_obs.append(next_obs)
                # next_obs = obs
                # next_obs[:,1]=obs[:,1]
                # next_obs[:,0]=obs[:,0]

                steps_added.append(len(obs))
                samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term}
                self._model_pool.add_samples(samples)

                nonterm_mask = ~term.squeeze(-1)
                if nonterm_mask.sum() == 0:
                    print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                    break

                obs = next_obs[nonterm_mask]


        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'mean_rollout_length': mean_rollout_length}
        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length, self._n_train_repeat
        ))
        return rollout_stats

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

    def _training_batch(self, batch_size=None):
        """
        returns batches, either only env_batch if real_ratio=1
        or 
        a dict where each env_batch and its associated model_batch are contained
        """
        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size*self._real_ratio)
        model_batch_size = batch_size - env_batch_size

        ## can sample from the env pool even if env_batch_size == 0
        env_batch = self._pool.random_batch(env_batch_size)

        if model_batch_size > 0:
            ### if real ratio > 0
            model_batch = self._model_pool.random_batch(model_batch_size)

            keys = env_batch.keys()
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch


    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                    "loss", "gradients", "gradient_norm", "global_gradient_norm"
                ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        self._training_progress.update()
        self._training_progress.set_description()

        feed_dict = self._get_feed_dict(iteration, batch)

        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        ### unstack for sac

        feed_dict = {
            self._observations_ph: batch['observations'][:,-self.active_obs_dim:],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'][:,-self.active_obs_dim:],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

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
