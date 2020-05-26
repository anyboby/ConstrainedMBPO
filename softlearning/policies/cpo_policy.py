"""Safety Policy"""

from collections import OrderedDict
from contextlib import contextmanager
import warnings


import numpy as np
import tensorflow as tf
from copy import deepcopy
import random 
import itertools

import softlearning.policies.safe_utils.trust_region as tro
from softlearning.policies.safe_utils.utils import values_as_sorted_list
from softlearning.policies.safe_utils.utils import EPS
from softlearning.policies.safe_utils.mpi_tools import mpi_avg, mpi_fork, proc_id, num_procs
from softlearning.policies.safe_utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from softlearning.models.ac_network import mlp_actor_critic, \
                                            mlp_actor, \
                                            mlp_critic, \
                                            get_vars, \
                                            count_vars, \
                                            placeholder, \
                                            placeholders, \
                                            placeholders_from_spaces
from softlearning.policies.safe_utils.logx import EpochLogger, setup_logger_kwargs, \
                                                Saver
from mbpo.models.bnn_chua import BNN
from mbpo.models.constructor import construct_model
from mbpo.models.fc import FC

import mbpo.models.nn as nn

from .base_policy import BasePolicy

class Agent:

    def __init__(self, **kwargs):
        self.params = deepcopy(kwargs)

    def set_logger(self, logger):
        self.logger = logger

    def prepare_update(self, training_package):
        # training_package is a dict with everything we need (and more)
        # to train.
        self.training_package = training_package

    def prepare_session(self, sess):
        self.sess = sess

    def update_pi(self, inputs):
        raise NotImplementedError

    def log(self):
        pass

    def ensure_satisfiable_penalty_use(self):
        reward_penalized = self.params.get('reward_penalized', False)
        objective_penalized = self.params.get('objective_penalized', False)
        assert not(reward_penalized and objective_penalized), \
            "Can only use either reward_penalized OR objective_penalized, " + \
            "not both."

        if not(reward_penalized or objective_penalized):
            learn_penalty = self.params.get('learn_penalty', False)
            assert not(learn_penalty), \
                "If you are not using a penalty coefficient, you should " + \
                "not try to learn one."

    def ensure_satisfiable_optimization(self):
        first_order = self.params.get('first_order', False)
        trust_region = self.params.get('trust_region', False)
        assert not(first_order and trust_region), \
            "Can only use either first_order OR trust_region, " + \
            "not both."

    @property
    def cares_about_cost(self):
        return self.use_penalty or self.constrained

    @property
    def clipped_adv(self):
        return self.params.get('clipped_adv', False)

    @property
    def constrained(self):
        return self.params.get('constrained', False)

    @property
    def first_order(self):
        self.ensure_satisfiable_optimization()
        return self.params.get('first_order', False)

    @property
    def learn_penalty(self):
        # Note: can only be true if "use_penalty" is also true.
        self.ensure_satisfiable_penalty_use()
        return self.params.get('learn_penalty', False)

    @property
    def penalty_param_loss(self):
        return self.params.get('penalty_param_loss', False)

    @property
    def objective_penalized(self):
        self.ensure_satisfiable_penalty_use()
        return self.params.get('objective_penalized', False)

    @property
    def reward_penalized(self):
        self.ensure_satisfiable_penalty_use()
        return self.params.get('reward_penalized', False)

    @property
    def save_penalty(self):
        # Essentially an override for CPO so it can save a penalty coefficient
        # derived in its inner-loop optimization process.
        return self.params.get('save_penalty', False)

    @property
    def trust_region(self):
        self.ensure_satisfiable_optimization()
        return self.params.get('trust_region', False)

    @property
    def use_penalty(self):
        return self.reward_penalized or \
               self.objective_penalized

class Safe_Policy(BasePolicy):
    def __init__(self,*args, **kwargs):
        super(Safe_Policy, self).__init__(*args,**kwargs)


class TrustRegionAgent(Agent):

    def __init__(self, damping_coeff=0.1, 
                       backtrack_coeff=0.8, 
                       backtrack_iters=10, 
                       **kwargs):
        super().__init__(**kwargs)
        self.damping_coeff = damping_coeff
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters
        self.params.update(dict(
            trust_region=True
            ))




#   reward_penalized=False,     # Irrelevant in CPO
#   objective_penalized=False,  # Irrelevant in CPO
#   learn_penalty=False,        # Irrelevant in CPO
#   penalty_param_loss=False    # Irrelevant in CPO


class CPOAgent(TrustRegionAgent):

    def __init__(self, learn_margin=False, **kwargs):
        super().__init__(**kwargs)
        self.learn_margin = learn_margin
        self.params.update(dict(
            constrained=True,
            save_penalty=True
            ))
        self.margin = 0
        self.margin_lr = 0.05
        self.margin_discount = 0.93
        self.c_gamma = kwargs['c_gamma']
        self.d_control = True
        self.delta = 0.6
        self.decayed_surr_cost = 0
        self.surr_cost_decay = 0.6
        self.max_path_length = kwargs['max_path_length']


    def update_pi(self, inputs):

        flat_g = self.training_package['flat_g']
        flat_b = self.training_package['flat_b']
        v_ph = self.training_package['v_ph']
        hvp = self.training_package['hvp']
        get_pi_params = self.training_package['get_pi_params']
        set_pi_params = self.training_package['set_pi_params']
        pi_loss = self.training_package['pi_loss']
        surr_cost = self.training_package['surr_cost']
        d_kl = self.training_package['d_kl']
        target_kl = self.training_package['target_kl']
        cost_lim = self.training_package['cost_lim']
        cur_cost_avg_op = self.training_package['cur_cost_avg']

        Hx = lambda x : mpi_avg(self.sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        outs = self.sess.run([flat_g, flat_b, pi_loss, surr_cost, cur_cost_avg_op], feed_dict=inputs)
        outs = [mpi_avg(out) for out in outs]
        g, b, pi_l_old, surr_cost_old, cur_cost_avg = outs

        # Need old params, old policy cost gap (epcost - limit), 
        # and surr_cost rescale factor (equal to average eplen).
        old_params = self.sess.run(get_pi_params)

        # calc the cost lim if it were discounted, 
        # need rescale since cost_lim refers to one episode
        #rescale = self.logger.get_stats('EpLen')[0]
        rescale = self.max_path_length
        cost_lim_disc = (cost_lim/rescale)/(1-self.c_gamma)

        # compare
        c_debug = self.logger.get_stats('CostEp')[0] - cost_lim
        c_debug /= (rescale + EPS)

        # undiscount (@anyboby TODO not really understand why we undiscount here, despite
        # the theory in CPO suggests the discounted Return)
        #c = rescale*(c_ret_old - cost_lim_disc)*(1-self.c_gamma)
        c = cur_cost_avg*rescale-cost_lim

        if self.d_control:
            c += self.delta * self.decayed_surr_cost

        # Consider the right margin
        if self.learn_margin:
            self.margin += self.margin_lr * c
            self.margin = max(0, self.margin)
            self.margin_lr *= self.margin_discount  # dampen margin lr to get asymptotic behavior

        # The margin should be the same across processes anyhow, but let's
        # mpi_avg it just to be 100% sure there's no drift. :)
        self.margin = mpi_avg(self.margin)

        # Adapt threshold with margin.
        c += self.margin

        # c + rescale * b^T (theta - theta_k) <= 0, equiv c/rescale + b^T(...)
        c /= (rescale + EPS)

        # Core calculations for CPO
        v = tro.cg(Hx, g)
        approx_g = Hx(v)
        q = np.dot(v, approx_g)

        # Determine optim_case (switch condition for calculation,
        # based on geometry of constrained optimization problem)
        if np.dot(b,b) <= 1e-8 and c < 0:
            # feasible and cost grad is zero---shortcut to pure TRPO update!
            w, r, s, A, B = 0, 0, 0, 0, 0
            optim_case = 4
        else:
            # cost grad is nonzero: CPO update!
            w = tro.cg(Hx, b)
            r = np.dot(w, approx_g)         # b^T H^{-1} g
            s = np.dot(w, Hx(w))            # b^T H^{-1} b
            A = q - r**2 / s                # should be always positive (Cauchy-Shwarz)
            B = 2*target_kl - c**2 / s      # does safety boundary intersect trust region? (positive = yes)

            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c < 0 and B >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c >= 0 and B >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
                self.logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                self.logger.log('Alert! Attempting infeasible recovery!', 'red')

        if optim_case in [3,4]:
            lam = np.sqrt(q / (2*target_kl))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, r /c], [r/c, np.inf]
            LA, LB = (LA, LB) if c < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(np.sqrt(A/B), LA)
            lam_b = proj(np.sqrt(q/(2*target_kl)), LB)
            f_a = lambda lam : -0.5 * (A / (lam+EPS) + B * lam) - r*c/(s+EPS)
            f_b = lambda lam : -0.5 * (q / (lam+EPS) + 2 * target_kl * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c - r) / (s + EPS)
        else:
            lam = 0
            nu = np.sqrt(2 * target_kl / (s+EPS))

        # normal step if optim_case > 0, but for optim_case =0,
        # perform infeasible recovery: step to purely decrease cost
        x = (1./(lam+EPS)) * (v + nu * w) if optim_case > 0 else nu * w

        # save intermediates for diagnostic purposes
        self.logger.store(Optim_A=A, Optim_B=B, Optim_c=c,
                          Optim_c_debug=c_debug, ###@anyboby TODO remove
                          Optim_q=q, Optim_r=r, Optim_s=s,
                          Optim_Lam=lam, Optim_Nu=nu, 
                          Penalty=nu, PenaltyDelta=0,
                          Margin=self.margin,
                          OptimCase=optim_case,
                          )

        def set_and_eval(step):
            self.sess.run(set_pi_params, feed_dict={v_ph: old_params - step * x})
            return mpi_avg(self.sess.run([d_kl, pi_loss, surr_cost], feed_dict=inputs))


        self.logger.log("cur_cost_limit %.3f"%cost_lim, "blue")

        # CPO uses backtracking linesearch to enforce constraints
        self.logger.log('surr_cost_old %.3f'%surr_cost_old, 'blue')
        for j in range(self.backtrack_iters):
            kl, pi_l_new, surr_cost_new = set_and_eval(step=self.backtrack_coeff**j)
            self.logger.log('%d \tkl %.3f \tsurr_cost_new %.3f'%(j, kl, surr_cost_new), 'blue')
            if (kl <= target_kl and
                (pi_l_new <= pi_l_old if optim_case > 1 else True) and
                surr_cost_new - surr_cost_old <= max(-c,0)):
                self.decayed_surr_cost = (surr_cost_new-surr_cost_old)+self.surr_cost_decay*self.decayed_surr_cost
                self.logger.log('Accepting new params at step %d of line search.'%j)
                self.logger.store(BacktrackIters=j)
                break

            if j==self.backtrack_iters-1:
                self.logger.log('Line search failed! Keeping old params.')
                self.logger.store(BacktrackIters=j)
                kl, pi_l_new, surr_cost_new = set_and_eval(step=0.)


    def log(self):
        self.logger.log_tabular('Optim_A', average_only=True)
        self.logger.log_tabular('Optim_B', average_only=True)
        self.logger.log_tabular('Optim_c', average_only=True)
        self.logger.log_tabular('Optim_c_debug', average_only=True) ### @anyboby TODO remove
        self.logger.log_tabular('Optim_q', average_only=True)
        self.logger.log_tabular('Optim_r', average_only=True)
        self.logger.log_tabular('Optim_s', average_only=True)
        self.logger.log_tabular('Optim_Lam', average_only=True)
        self.logger.log_tabular('Optim_Nu', average_only=True)
        self.logger.log_tabular('OptimCase', average_only=True)
        self.logger.log_tabular('Margin', average_only=True)
        self.logger.log_tabular('BacktrackIters', average_only=True)

class CPOPolicy(BasePolicy):
    def __init__(self, 
                obs_space, 
                act_space, 
                session,
                logger=None,
                *args, **kwargs,
                ):

        # ___________________________________________ #
        #                   Params                    #
        # ___________________________________________ #
        self.vf_lr = kwargs.get('vf_lr', 3e-4)
        self.cvf_lr = kwargs.get('cvf_lr', 1e-3)
        self.target_kl = kwargs.get('target_kl', 0.01)
        self.ent_reg = kwargs.get('ent_reg', 0.01)
        self.cost_lim_end = kwargs.get('cost_lim_end', 25)
        self.cost_lim = kwargs.get('cost_lim', 25)
        self.cost_lam = kwargs.get('cost_lam', 0.97)
        self.cost_gamma = kwargs.get('cost_gamma', 0.99)
        self.lam = kwargs.get('lam', 0.97)
        self.gamma = kwargs.get('discount', 0.99)
        self.vf_iters = kwargs.get('vf_iters', 80)
        self.max_path_length = kwargs.get('max_path_length', 1)
        self.critic_ensemble_size = kwargs.get('critic_ensemble_size', 1)
        #usually not deterministic, but give the option for eval runs
        self._deterministic = False
        
        
        cpo_kwargs = dict(  reward_penalized=False,  # Irrelevant in CPO
                    objective_penalized=False,  # Irrelevant in CPO
                    learn_penalty=False,  # Irrelevant in CPO
                    penalty_param_loss=False,  # Irrelevant in CPO
                    learn_margin=False, #learn_margin=True,
                    c_gamma = self.cost_gamma,
                    max_path_length=self.max_path_length
                    )

        # ________________________________ #        
        #       Cpo agent and logger       #
        # ________________________________ #

        log_dir = kwargs.get('log_dir','~/ray_cmbpo/')
        self.agent = CPOAgent(**cpo_kwargs)
        exp_name = 'cpo'
        test_seed = random.randint(0,9999)
        #logger_kwargs = setup_logger_kwargs(exp_name, test_seed, data_dir=log_dir)
        if logger:
            self.logger = logger
        else:
            self.logger = EpochLogger()
            #self.logger.save_config(locals())
            self.agent.set_logger(self.logger)
        
        self.saver = Saver()
        self.sess = session
        self.agent.prepare_session(self.sess)
        self.act_space = act_space
        self.obs_space = obs_space
        self.ep_len = kwargs.get('epoch_length')

        # ___________________________________________ #
        #              Prepare ac network             #
        # ___________________________________________ #
        scope = 'policy'
        with tf.variable_scope(scope):
            # kwargs for ac network
            a_kwargs=dict()
            a_kwargs['action_space'] = self.act_space
            a_kwargs['hidden_sizes'] = kwargs.get('hidden_layer_sizes_a')

            c_kwargs = dict()
            c_kwargs['hidden_sizes'] = kwargs.get('hidden_layer_sizes_c')

            # tf placeholders
            with tf.variable_scope('obs_ph'):
                self.obs_ph = placeholders_from_spaces(self.obs_space)[0]
            with tf.variable_scope('a_ph'): 
                self.a_ph = placeholders_from_spaces(self.act_space)[0]

            # input placeholders to computation graph for batch data
            with tf.variable_scope('adv_ph'):
                self.adv_ph = placeholder(None)
            with tf.variable_scope('cadv_ph'):
                self.cadv_ph = placeholder(None)
            with tf.variable_scope('logp_old_ph'):
                self.logp_old_ph = placeholder(None)
            with tf.variable_scope('surr_cost_rescale_ph'):
                # phs for cpo specific inputs to comp graph
                self.surr_cost_rescale_ph = placeholder(None)
            with tf.variable_scope('cur_cost_ph'):
                self.cur_cost_ph = placeholder(None)

            # critic phs
            with tf.variable_scope('ret_ph'):
                self.ret_ph = placeholder(None)
            with tf.variable_scope('cret_ph'):
                self.cret_ph = placeholder(None)
            with tf.variable_scope('old_v_ph'):
                self.old_v_ph = placeholder(None)
            with tf.variable_scope('old_vc_ph'):
                self.old_vc_ph = placeholder(None)

            self.actor = mlp_actor

            actor_outs = self.actor(self.obs_ph, self.a_ph, **a_kwargs)
            self.pi, self.logp, self.logp_pi, self.pi_info, self.pi_info_phs, self.d_kl, self.ent \
                = actor_outs

            
            self.v1, self.vc1 = mlp_critic(self.obs_ph, name='v1', **c_kwargs)
            self.v1_mb, self.vc1_mb = mlp_critic(self.obs_ph, name='v1_mb', **c_kwargs)
            self.v2_mb, self.vc2_mb = mlp_critic(self.obs_ph, name='v2_mb', **c_kwargs)

            self.v2_cl, self.vc2_cl = mlp_critic(self.obs_ph, name='v_cl', **c_kwargs)

            self.v = construct_model(in_dim=np.prod(self.obs_space.shape),
                        out_dim=1,
                        name='VEnsemble',
                        hidden_dims=kwargs.get('hidden_layer_sizes_c'),
                        lr=self.vf_lr/15, 
                        num_networks=self.critic_ensemble_size, 
                        activation='ReLU',
                        loss='MSE',
                        num_elites=5,
                        session=self.sess)

            self.vc_relu = construct_model(in_dim=np.prod(self.obs_space.shape),
                        out_dim=1,
                        name='VCEnsemble_Relu',
                        hidden_dims=kwargs.get('hidden_layer_sizes_c'),
                        lr=self.cvf_lr, 
                        num_networks=self.critic_ensemble_size, 
                        activation='ReLU',
                        loss='MSE',
                        num_elites=5,
                        session=self.sess)
	
            self.vc_swish = construct_model(in_dim=np.prod(self.obs_space.shape),
                        out_dim=1,
                        name='VCEnsemble_Swish',
                        hidden_dims=kwargs.get('hidden_layer_sizes_c'),
                        lr=self.cvf_lr/15, 
                        num_networks=self.critic_ensemble_size, 
                        activation='swish',
                        loss='MSE',
                        num_elites=5,
                        session=self.sess)

            self.vc_mmb = construct_model(in_dim=np.prod(self.obs_space.shape),
                        out_dim=1,
                        name='VCEnsemble_mmb',
                        hidden_dims=kwargs.get('hidden_layer_sizes_c'),
                        lr=self.cvf_lr/15, 
                        num_networks=self.critic_ensemble_size, 
                        activation='swish',
                        loss='MSE',
                        num_elites=5,
                        session=self.sess)
            
            self.vc_mmb_lr20 = construct_model(in_dim=np.prod(self.obs_space.shape),
                        out_dim=1,
                        name='VCEnsemble_mmb_lr20',
                        hidden_dims=kwargs.get('hidden_layer_sizes_c'),
                        lr=self.cvf_lr/20, 
                        num_networks=self.critic_ensemble_size, 
                        activation='swish',
                        loss='MSE',
                        num_elites=5,
                        session=self.sess)

            self.vc_mmb_lr15_epadap = construct_model(in_dim=np.prod(self.obs_space.shape),
                        out_dim=1,
                        name='VCEnsemble_mmb_lr15_epadap',
                        hidden_dims=kwargs.get('hidden_layer_sizes_c'),
                        lr=self.cvf_lr/15, 
                        num_networks=self.critic_ensemble_size, 
                        activation='swish',
                        loss='MSE',
                        num_elites=5,
                        session=self.sess)

            ######################### DEBUG !!! ###########################
            params = {'name': 'vc_debug', 
                        'num_networks': 7, 
                        'sess': self.sess}       
            self.vc_debug = BNN(params)
            self.vc_debug.add(FC(kwargs.get('hidden_layer_sizes_c')[0], input_dim=np.prod(self.obs_space.shape), activation='swish', weight_decay=0.0000))	#0.000025))
            
            for hidden_dim in kwargs.get('hidden_layer_sizes_c')[1:]:
                self.vc_debug.add(FC(hidden_dim, activation='swish', weight_decay=0.0000))			#0.00005))
            
            self.vc_debug.add(FC(1, weight_decay=0.000))
            opt_params = {"learning_rate":self.cvf_lr/15} 
            self.vc_debug.finalize(tf.train.AdamOptimizer, opt_params)

            self.vc_st  = nn.EnsembleFeedForwardNet(
                'vc_st', 
                in_size= np.prod(self.obs_space.shape), 
                out_shape = [],
                layers=4,
                hidden_dim=128,
                get_uncertainty=True,
                ensemble_size=4,
                train_sample_count=4,
                eval_sample_count=4,
                )            
            self.vc_st_guess = self.vc_st(self.obs_ph, is_eval=False, reduce_mode="random")
            self.vc_st_loss = .5 * tf.reduce_mean(tf.square(self.vc_st_guess-self.cret_ph))
            st_opt = tf.train.AdamOptimizer(self.cvf_lr)
            self.vc_st_trainop = st_opt.minimize(self.vc_st_loss)

            ######## classic v loss ########
            self.v_cla_loss = .5* tf.reduce_mean(tf.square(self.ret_ph-self.v1))
            self.vc_cla_loss = .5*tf.reduce_mean(tf.square(self.cret_ph-self.vc1))
            self.train_v_cla = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_cla_loss)
            self.train_vc_cla = MpiAdamOptimizer(learning_rate=self.cvf_lr).minimize(self.vc_cla_loss)
            ###############################

            ######## classic v mb ##########
            self.v_cla_mb_loss = .5* tf.reduce_mean(tf.square(self.ret_ph-self.v1_mb))
            self.vc_cla_mb_loss = .5*tf.reduce_mean(tf.square(self.cret_ph-self.vc1_mb))
            self.train_v_cla_mb = MpiAdamOptimizer(learning_rate=self.vf_lr/15).minimize(self.v_cla_mb_loss)
            self.train_vc_cla_mb = MpiAdamOptimizer(learning_rate=self.cvf_lr/15).minimize(self.vc_cla_mb_loss)

            self.v2_cla_mb_loss = .5* tf.reduce_mean(tf.square(self.ret_ph-self.v2_mb))
            self.vc2_cla_mb_loss = .5*tf.reduce_mean(tf.square(self.cret_ph-self.vc2_mb))
            self.train_v2_cla_mb = MpiAdamOptimizer(learning_rate=self.vf_lr/10).minimize(self.v2_cla_mb_loss)
            self.train_vc2_cla_mb = MpiAdamOptimizer(learning_rate=self.cvf_lr/10).minimize(self.vc2_cla_mb_loss)

            ######## classic v mb end ##########
            
            ######## clipped loss #########
            # @anyboby testing value clipping
            old_vpred = self.old_v_ph
            v_cliprange = 0.1
            v_clipped = old_vpred + tf.clip_by_value(self.v2_cl-old_vpred, -v_cliprange, v_cliprange)
            v_loss1 = tf.square(self.ret_ph-self.v2_cl)
            v_loss2 = tf.square(self.ret_ph-v_clipped)
            self.v_loss_cl = .5*tf.reduce_mean(tf.minimum(v_loss1, v_loss2))
            self.v_loss_cl_mse = .5 * tf.reduce_mean(v_loss1)
            #self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

            old_vcpred = self.old_v_ph
            
            vc_cliprange = 0.2 # tf.reduce_mean(old_vcpred/100)    #vc_cliprange = 0.2, experimental automatic setup !
            vc_clipped = old_vcpred + tf.clip_by_value(self.vc2_cl-old_vcpred, -vc_cliprange, vc_cliprange)
            vc_loss1 = tf.square(self.cret_ph-self.vc2_cl)

            vc_loss2 = tf.square(self.cret_ph-vc_clipped)
            self.vc_loss_cl = .5*tf.reduce_mean(tf.minimum(vc_loss1, vc_loss2))
            self.vc_loss_cl_mse = .5*tf.reduce_mean(vc_loss1)
            #self.vc_loss = tf.reduce_mean((self.cret_ph - self.vc)**2)

            # If agent uses penalty directly in reward function, don't train a separate
            # value function for predicting cost returns. (Only use one vf for r - p*c.)
            total_value_loss = self.v_loss_cl + self.vc_loss_cl

            # Optimizer for value learning
            #self.train_vf = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(total_value_loss)
            #@anyboby testing: this shouldn't make a difference, but lets try it:
            self.train_vc_cl = MpiAdamOptimizer(learning_rate=self.cvf_lr).minimize(self.vc_loss_cl)
            self.train_v_cl = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss_cl)

            #################################################################

            # Organize placeholders for zipping with data from buffer on updates
            self.buf_fields = [self.obs_ph, self.a_ph, self.adv_ph,
                                self.cadv_ph, 'returns', 'creturns',
                                self.logp_old_ph, self.old_v_ph, self.old_vc_ph,
                                self.cur_cost_ph]
            self.buf_fields += values_as_sorted_list(self.pi_info_phs)

            self.actor_phs = [self.obs_ph, self.a_ph, self.adv_ph,
                                self.cadv_ph, self.logp_old_ph,
                                self.cur_cost_ph]
            self.actor_phs += values_as_sorted_list(self.pi_info_phs)
            
            self.critic_phs = [
                self.obs_ph, self.old_v_ph, self.old_vc_ph
            ]

            # organize tf ops required for generation of actions
            self.ops_for_action = dict(pi=self.pi, 
                                logp_pi=self.logp_pi,
                                pi_info=self.pi_info)

            # organize tf ops for diagnostics
            self.ops_for_diagnostics = dict(pi=self.pi,
                                            logp_pi=self.logp_pi,
                                            pi_info=self.pi_info,
                                            )

            # Count variables
            var_counts = tuple(count_vars(scope) for scope in ['pi', 'VCEnsemble', 'VEnsemble'])
            self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)

            # Make a sample estimate for entropy to use as sanity check
            approx_ent = tf.reduce_mean(-self.logp)

            # @anyboby borrowed from sac maybe for later ######
            target_entropy = kwargs['target_entropy']
            self._target_entropy = (
                -np.prod(self.act_space.shape)
                if target_entropy == 'auto'
                else target_entropy)

            self.ent_reg=0.01
            ##### ---------------- #####

            # ________________________________ #        
            #    Computation graph for policy  #
            # ________________________________ #
            ratio = tf.exp(self.logp-self.logp_old_ph)
            # Surrogate advantage / clipped surrogate advantage
            surr_adv = tf.reduce_mean(ratio * self.adv_ph)

            # Surrogate cost (advantage)
            self.surr_cost = tf.reduce_mean(ratio * self.cadv_ph)

            # Cret
            self.cur_cost_avg = tf.reduce_mean(self.cur_cost_ph)

            # cost_lim if it were discounted
            # self.disc_cost_lim = (self.cost_lim/self.ep_len)

            # Create policy objective function, including entropy regularization
            pi_objective = surr_adv + self.ent_reg * self.ent

            # Loss function for pi is negative of pi_objective
            self.pi_loss = -pi_objective

            # Optimizer-specific symbols
            if self.agent.trust_region:              ### <------- CPO
                            # Symbols needed for CG solver for any trust region method
                pi_params = get_vars('pi')
                flat_g = tro.flat_grad(self.pi_loss, pi_params)
                v_ph, hvp = tro.hessian_vector_product(self.d_kl, pi_params)
                if self.agent.damping_coeff > 0:
                    hvp += self.agent.damping_coeff * v_ph

                # Symbols needed for CG solver for CPO only
                flat_b = tro.flat_grad(self.surr_cost, pi_params)

                # Symbols for getting and setting params
                get_pi_params = tro.flat_concat(pi_params)
                set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)

                self.training_package = dict(flat_g=flat_g,
                                        flat_b=flat_b,
                                        v_ph=v_ph,
                                        hvp=hvp,
                                        get_pi_params=get_pi_params,
                                        set_pi_params=set_pi_params)

            else:
                raise NotImplementedError

            # Provide training package to agent
            self.training_package.update(dict(pi_loss=self.pi_loss, 
                                        surr_cost=self.surr_cost,
                                        cur_cost_avg = self.cur_cost_avg,
                                        d_kl=self.d_kl, 
                                        target_kl=self.target_kl,
                                        cost_lim=self.cost_lim))
            self.agent.prepare_update(self.training_package)

        ##### set up saver after all graph building is done
        self.saver.init_saver(scope=scope)



    def shuffle_rows(self, arr):
        idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
        return arr[np.arange(arr.shape[0])[:, None], idxs]


    def set_logger(self, logger):
        """
        provide a logger (Policy creates it's own logger by default, 
        but you might want to share a logger between algo, samplers, etc.)
        
        automatically shares logger with agent
        Args: 
            logger : instance of EpochLogger
        """ 
        self.logger = logger
        self.agent.set_logger(logger) #share logger with agent

    #@anyboby todo: buf_inputs has to be delivered to update. implement when buffer (or pool) is done
    def update(self, buf_inputs, ):
        cur_cost = self.logger.get_stats('CostEp')[0]
        #cur_cost_lim = self.cost_lim-self._epoch*(self.cost_lim-self.cost_lim_end)/self._n_epochs + random.randint(0, rand_cost)
        cur_cost_lim = self.cost_lim
        c = cur_cost - cur_cost_lim
        if c > 0 and self.agent.cares_about_cost:
            self.logger.log('Warning! Safety constraint is already violated.', 'red')

        #=====================================================================#
        #  Prepare feed dict                                                  #
        #=====================================================================#

        inputs = {k:v for k,v in zip(self.buf_fields, buf_inputs)}
        actor_inputs = {buf_ph:inputs[buf_ph] for buf_ph in self.actor_phs}
        critic_inputs = {buf_ph:inputs[buf_ph] for buf_ph in self.critic_phs}
        critic_inputs[self.ret_ph] = inputs['returns']
        critic_inputs[self.cret_ph] = inputs['creturns']
        #=====================================================================#
        #  Make some measurements before updating                             #
        #=====================================================================#

        measures = dict(LossPi=self.pi_loss,
                        SurrCost=self.surr_cost,
                        Entropy=self.ent)


        pre_update_measures = self.sess.run(measures, feed_dict=actor_inputs)
        
        critic_diag = self.compute_v_losses(inputs)
        pre_update_measures.update(critic_diag)

        self.logger.store(**pre_update_measures)

        #=====================================================================#
        #  update cost_limit (@mo creation)                               #
        #=====================================================================#
        # Provide training package to agent
        self.training_package["cost_lim"]= cur_cost_lim
        self.agent.prepare_update(self.training_package)

        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#
        self.agent.update_pi(actor_inputs)

        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#
        self.train_critic(
            critic_inputs, 
            batch_size=64, 
            min_epoch_before_break=int(self.vf_iters/5), 
            max_epochs=int(self.vf_iters/5), 
            holdout_ratio=0.1
            )

        #=====================================================================#
        #  Make some measurements after updating                              #
        #=====================================================================#

        del measures['Entropy']
        measures['KL'] = self.d_kl

        post_update_measures = self.sess.run(measures, feed_dict=actor_inputs)

        critic_diag_post = self.compute_v_losses(inputs)
        post_update_measures.update(critic_diag_post)

        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas[k+'Delta'] = post_update_measures[k] - pre_update_measures[k]
        self.logger.store(KL=post_update_measures['KL'], **deltas)

    def train_critic(self, inputs, **kwargs):
        obs_in = inputs[self.obs_ph]
        vc_targets = inputs[self.cret_ph][:, np.newaxis]
        v_targets = inputs[self.ret_ph][:, np.newaxis]

        batch_size = kwargs['batch_size']
        epochs = kwargs['min_epoch_before_break']
        holdout_ratio = kwargs['holdout_ratio']
        self.old_inputs = obs_in
        self.old_targets = vc_targets
        vc_metrics = self.vc_relu.train(obs_in,
                                    vc_targets,
                                    batch_size=len(obs_in),
                                    min_epoch_before_break = self.vf_iters,
                                    max_epochs= self.vf_iters,
                                    holdout_ratio= 0,
                                    )                                            
        _ = self.vc_swish.train(obs_in,
                                    vc_targets,
                                    **kwargs,
                                    )                                            
        _ = self.vc_mmb.train(obs_in,
                                    vc_targets,
                                    batch_size=int(batch_size/2),
                                    min_epoch_before_break = epochs,
                                    max_epochs= epochs,
                                    holdout_ratio= holdout_ratio,
                                    )                                            
        _ = self.vc_mmb_lr20.train(obs_in,
                                    vc_targets,
                                    batch_size=int(batch_size/2),
                                    min_epoch_before_break = epochs,
                                    max_epochs= epochs,
                                    holdout_ratio= holdout_ratio,
                                    )                                            
        _ = self.vc_mmb_lr15_epadap.train(obs_in,
                                    vc_targets,
                                    batch_size=int(batch_size/2),
                                    min_epoch_before_break = int(epochs*2),
                                    max_epochs = int(epochs*4),
                                    holdout_ratio= holdout_ratio,
                                    )                                                                                

        v_metrics = self.v.train(obs_in, 
                                    v_targets, 
                                    **kwargs,
                                    )
        self.vc_debug.train(obs_in,
                                    vc_targets,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    )                                
        ### mini_batch ###
        for i in range(epochs):
            idxs = np.random.choice(inputs[self.obs_ph].shape[0], size=[inputs[self.obs_ph].shape[0]], replace=False)

            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[batch_num * batch_size:(batch_num + 1) * batch_size]
                mb_inputs = {
                    self.obs_ph:inputs[self.obs_ph][batch_idxs],
                    self.ret_ph:inputs[self.ret_ph][batch_idxs],
                    self.cret_ph:inputs[self.cret_ph][batch_idxs],
                }
                self.sess.run(
                    self.train_vc_cla_mb,
                    feed_dict=mb_inputs
                )
                self.sess.run(
                    self.train_vc2_cla_mb,
                    feed_dict=mb_inputs
                )
        
        for i in range(self.vf_iters):
            self.sess.run(self.vc_st_trainop, feed_dict=inputs)
            self.sess.run(
                self.train_vc_cla, 
                feed_dict=inputs
                )
            self.sess.run(
                self.train_vc_cl, 
                feed_dict=inputs
                )



        return v_metrics.update(vc_metrics)

    def compute_v_losses(self, inputs):
        critic_inputs = {buf_ph:inputs[buf_ph] for buf_ph in self.critic_phs}
        critic_inputs[self.ret_ph] = inputs['returns']
        critic_inputs[self.cret_ph] = inputs['creturns']
        v_ins, v_tars, vc_ins, vc_tars = inputs[self.obs_ph], \
                                            inputs['returns'][:, np.newaxis], \
                                            inputs[self.obs_ph], \
                                            inputs['creturns'][:, np.newaxis]

        ################################## DEBUG ########################

        vloss = self.v.validate(v_ins, v_tars)
        vc_relu_loss = self.vc_relu.validate(vc_ins, vc_tars)        
        vc_swish_loss = self.vc_swish.validate(vc_ins, vc_tars)        
        vc_mmb_loss = self.vc_mmb.validate(vc_ins, vc_tars)        
        vc_mmb_lr20_loss = self.vc_mmb_lr20.validate(vc_ins, vc_tars)        
        vc_mmb_lr15_epadap_loss = self.vc_mmb_lr15_epadap.validate(vc_ins, vc_tars)                                

        vc_ins_deb = np.tile(v_ins[np.newaxis,...], (7,1,1))
        vc_outs_deb = np.tile(v_tars[np.newaxis,...], (7,1,1))
        vc_deb_loss = self.sess.run(self.vc_debug.mse_loss, feed_dict={self.vc_debug.sy_train_in:vc_ins_deb, self.vc_debug.sy_train_targ: vc_outs_deb})
        vc_st_loss = self.sess.run(self.vc_st_loss, feed_dict=critic_inputs)
        vc_cla_loss = self.sess.run(
            self.vc_cla_loss, 
            feed_dict=critic_inputs
        )

        vc_cla_loss_mb = self.sess.run(
            self.vc_cla_mb_loss, 
            feed_dict=critic_inputs
        )

        vc2_cla_loss_mb = self.sess.run(
            self.vc2_cla_mb_loss, 
            feed_dict=critic_inputs
        )

        vc_cl_loss = self.sess.run(
            self.vc_loss_cl, 
            feed_dict=critic_inputs
        )
        vc_cl_loss_mse = self.sess.run(
            self.vc_loss_cl_mse, 
            feed_dict=critic_inputs
        )

        critic_metric = dict()
        critic_metric['LossV'] = vloss
        critic_metric['LossVC_ReLu'] = vc_relu_loss
        critic_metric['LossVC_Swish'] = vc_swish_loss
        critic_metric['LossVC_minminb'] = vc_mmb_loss
        critic_metric['LossVC_minminb_lr20'] = vc_mmb_lr20_loss
        critic_metric['LossVC_minminb_lr15_epadap'] = vc_mmb_lr15_epadap_loss
        critic_metric['LossVC_chua'] = vc_deb_loss.mean()
        critic_metric['LossVC_STEVE'] = vc_st_loss
        critic_metric['LossVC_classic'] = vc_cla_loss
        critic_metric['LossVC_clipped_mse'] = vc_cl_loss_mse
        critic_metric['LossVC_clipped'] = vc_cl_loss
        critic_metric['LossVC_classic_mb1'] = vc_cla_loss_mb
        critic_metric['LossVC_classic_mb2'] = vc2_cla_loss_mb
        ################################## DEBUG END ########################

        return critic_metric

    def format_obs_for_tf(self, obs):
        sq_obs = np.squeeze(obs)
        if len(sq_obs.shape) == len(self.obs_space.shape):
            sq_obs = sq_obs[np.newaxis]
        elif len(sq_obs.shape) > len(self.obs_space.shape):
            sq_obs = sq_obs
        else: 
            raise Exception('faulty obs')
        return sq_obs

    def reset(self):
        pass

    def actions(self, obs):
        get_action_outs = self.get_action_outs(obs)
        a = get_action_outs['pi']
        return a

    def actions_np(self, obs):
        actions = self.actions(obs)
        return np.array(actions)

    def log_pis(self, obs, a):
        pass

    def get_action_outs(self, obs):
        # check if single obs or multiple
        # remove single dims
        feed_obs = self.format_obs_for_tf(obs)
        get_action_outs = self.sess.run(self.ops_for_action, 
                        feed_dict={self.obs_ph: feed_obs})
        v = self.sess.run(self.v1_mb, 
                        feed_dict={self.obs_ph: feed_obs})
        vc = self.sess.run(self.vc1_mb, 
                        feed_dict={self.obs_ph: feed_obs})

        # v = np.squeeze(self.v.predict(feed_obs, factored=True), axis=-1)
        # vc = np.squeeze(self.vc_relu.predict(feed_obs, factored=True), axis=-1)
        get_action_outs['v'] = v
        get_action_outs['vc'] = vc
        return get_action_outs

    def get_v(self, obs):
        feed_obs = self.format_obs_for_tf(obs)
        v = self.sess.run(self.v1_mb, 
                        feed_dict={self.obs_ph: feed_obs})
        # v = np.squeeze(self.v.predict(feed_obs, factored=True), axis=-1)
        return v

    def get_vc(self, obs):
        feed_obs = self.format_obs_for_tf(obs)
        vc = self.sess.run(self.vc1_mb, 
                        feed_dict={self.obs_ph: feed_obs})

        # vc = np.squeeze(self.vc_relu.predict(feed_obs, factored=True), axis=-1)
        return vc

    @contextmanager
    def set_deterministic(self, deterministic=True):
        """Context manager for changing the determinism of the policy.
        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
        """
        warnings.warn('Deterministic CPOPolicy not implemented, has no effect')

        was_deterministic = self._deterministic
        self._deterministic = deterministic
        yield
        self._deterministic = was_deterministic

    def log_pis_np(self, obs, a):
        pass

    def get_weights(self):
        return get_vars()

    def set_weights(self, *args, **kwargs):
        raise NotImplementedError
        # return self.ac_network.set_weights(*args, **kwargs)
        
    @property
    def trainable_variables(self):
        return get_vars()

    @property
    def non_trainable_weights(self):
        raise NotImplementedError
        # """Due to our nested model structure, we need to filter duplicates."""
        # return list(set(super(CPOPolicy, self).non_trainable_weights))

    def save(self, checkpoint_dir):
        tf_path, model_info_path, success = self.saver.save_tf(self.sess, inputs={'x':self.obs_ph},
                        outputs={'pi':self.pi, 'v':self.v, 'vc':self.vc_relu}, 
                        output_dir=checkpoint_dir)
        return tf_path, model_info_path, success

    def log(self):
        logger = self.logger
        self.agent.log()


        # V loss and change
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossVDelta', average_only=True)
        
        # Vc loss and change, if applicable (reward_penalized agents don't use vc)
        if not(self.agent.reward_penalized):
            logger.log_tabular('LossVC_ReLu', average_only=True)
            logger.log_tabular('LossVC_Swish', average_only=True)
            logger.log_tabular('LossVC_minminb', average_only=True)
            logger.log_tabular('LossVC_minminb_lr20', average_only=True)
            logger.log_tabular('LossVC_minminb_lr15_epadap', average_only=True)
            logger.log_tabular('LossVC_ReLuDelta', average_only=True)

            ##### DEBUG logs #####
            logger.log_tabular('LossVC_chua', average_only=True)
            logger.log_tabular('LossVC_STEVE', average_only=True)
            logger.log_tabular('LossVC_classic', average_only=True)
            logger.log_tabular('LossVC_clipped_mse', average_only=True)
            logger.log_tabular('LossVC_clipped', average_only=True)
            logger.log_tabular('LossVC_classic_mb1', average_only=True)
            logger.log_tabular('LossVC_classic_mb2', average_only=True)
            ##### DEBUG logs end #####

        if self.agent.use_penalty or self.agent.save_penalty:
            logger.log_tabular('Penalty', average_only=True)
            logger.log_tabular('PenaltyDelta', average_only=True)
        else:
            logger.log_tabular('Penalty', 0)
            logger.log_tabular('PenaltyDelta', 0)

        # Surr cost and change
        logger.log_tabular('SurrCost', average_only=True)
        logger.log_tabular('SurrCostDelta', average_only=True)

        # Policy stats
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)

        # Pi loss and change
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossPiDelta', average_only=True)


    def get_diagnostics(self, obs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """

        # check if single obs or multiple
        # remove single dims
        feed_obs = self.format_obs_for_tf(obs)

        get_diag_outs = self.sess.run(self.ops_for_diagnostics, 
                        feed_dict={self.obs_ph: feed_obs})
        

        a = get_diag_outs['pi']
        logp = get_diag_outs['logp_pi']
        pi_info = get_diag_outs['pi_info']

        v = self.get_v(obs)
        vc = self.get_vc(obs)  # Agent may not use cost value func


        pi_info_means = {'cpo/pi_info_'+k:np.mean(v) for k,v in pi_info.items()}
        diag = OrderedDict({
            'cpo/a-mean'        : np.mean(a),
            'cpo/a-std'         : np.std(a),
            'cpo/v-mean'      : np.mean(v),
            'cpo/v-std'       : np.std(v),
            'cpo/vc-mean'     : np.mean(vc),
            'cpo/vc-std'      : np.std(vc),
            'cpo/logp-mean'   : np.mean(logp),
            'cpo/logp-std'    : np.std(logp),
            'cpo/pi_info'     : pi_info,
            # 'd_kl-mean-std' : np.mean(d_kl),
            # 'd_kl-std'      : np.std(d_kl),
            # 'ent-mean'      : np.mean(ent),
            # 'ent-std'       : np.std(ent),
        })
        diag.update(pi_info_means)

        return diag
