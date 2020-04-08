"""Safety Policy"""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
from copy import deepcopy
import random 

import softlearning.policies.safe_utils.trust_region as tro
from softlearning.policies.safe_utils.utils import values_as_sorted_list
from softlearning.policies.safe_utils.utils import EPS
from softlearning.policies.safe_utils.mpi_tools import mpi_avg, mpi_fork, proc_id, num_procs
from softlearning.policies.safe_utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from softlearning.models.ac_network import mlp_actor_critic, \
                                            get_vars, \
                                            count_vars, \
                                            placeholders, \
                                            placeholders_from_spaces
from softlearning.policies.safe_utils.logx import EpochLogger, setup_logger_kwargs

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





class CPOAgent(TrustRegionAgent):

#   reward_penalized=False,     # Irrelevant in CPO
#   objective_penalized=False,  # Irrelevant in CPO
#   learn_penalty=False,        # Irrelevant in CPO
#   penalty_param_loss=False    # Irrelevant in CPO

    def __init__(self, learn_margin=False, **kwargs):
        super().__init__(**kwargs)
        self.learn_margin = learn_margin
        self.params.update(dict(
            constrained=True,
            save_penalty=True
            ))
        self.margin = 0
        self.margin_lr = 0.05


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

        Hx = lambda x : mpi_avg(self.sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        outs = self.sess.run([flat_g, flat_b, pi_loss, surr_cost], feed_dict=inputs)
        outs = [mpi_avg(out) for out in outs]
        g, b, pi_l_old, surr_cost_old = outs

        # Need old params, old policy cost gap (epcost - limit), 
        # and surr_cost rescale factor (equal to average eplen).
        old_params = self.sess.run(get_pi_params)
        c = self.logger.get_stats('EpCost')[0] - cost_lim
        rescale = self.logger.get_stats('EpLen')[0]

        # Consider the right margin
        if self.learn_margin:
            self.margin += self.margin_lr * c
            self.margin = max(0, self.margin)

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
                          Optim_q=q, Optim_r=r, Optim_s=s,
                          Optim_Lam=lam, Optim_Nu=nu, 
                          Penalty=nu, DeltaPenalty=0,
                          Margin=self.margin,
                          OptimCase=optim_case)

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
                *args, **kwargs):
        


        cpo_kwargs = dict(  reward_penalized=False,  # Irrelevant in CPO
                    objective_penalized=False,  # Irrelevant in CPO
                    learn_penalty=False,  # Irrelevant in CPO
                    penalty_param_loss=False  # Irrelevant in CPO
                    )

        # ________________________________ #        
        #       Cpo agent and logger       #
        # ________________________________ #

        self.agent = CPOAgent(**cpo_kwargs)
        exp_name = 'cpo'
        test_seed = random.randint(0,9999)
        logger_kwargs = setup_logger_kwargs(exp_name, test_seed)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.agent.set_logger(self.logger)

        self.ac = mlp_actor_critic
        self.act_space = act_space
        self.obs_space = obs_space


        # ___________________________________________ #
        #                   Params                    #
        # ___________________________________________ #
        self.vf_lr = kwargs.get('vf_lr', 1e-3)
        self.target_kl = kwargs.get('target_kl', 0.01)
        self.cost_lim_end = kwargs.get('cost_lim_end', 25)
        self.cost_lim = kwargs.get('cost_lim', 25)
        self.cost_lam = kwargs.get('cost_lam', 0.97)
        self.cost_gamma = kwargs.get('cost_gamma', 0.99)
        self.lam = kwargs.get('lam', 0.97)
        self.gamma = kwargs.get('discount', 0.99)
        self.stepchs_per_epoch = kwargs.get('rollout_batch_size', 10000)
        self.vf_iters = kwargs.get('vf_iters', 80)

        
        
        # ___________________________________________ #
        #              Prepare ac network             #
        # ___________________________________________ #
        
        # kwargs for ac network
        ac_kwargs=dict()
        ac_kwargs['action_space'] = self.act_space
        ac_kwargs['hidden_sizes'] = kwargs['hidden_layer_sizes']
        # tf placeholders
        self.obs_ph, self.a_ph = placeholders_from_spaces(self.obs_space, self.act_space)

        # inputs to computation graph for batch data
        self.adv_ph, self.cadv_ph, self.ret_ph, self.cret_ph, self.logp_old_ph = placeholders(*(None for _ in range(5)))

        # phs for cpo specific inputs to comp graph
        self.surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
        self.cur_cost_ph = tf.placeholder(tf.float32, shape=())

        # unpack actor critic outputs
        ac_outs = self.ac(self.obs_ph, self.a_ph, **ac_kwargs)
        pi, logp, logp_pi, pi_info, pi_info_phs, self.d_kl, self.ent, v, vc = ac_outs

        # Organize placeholders for zipping with data from buffer on updates
        self.buf_phs = [self.obs_ph, self.a_ph, self.adv_ph, self.cadv_ph, self.ret_ph, self.cret_ph, self.logp_old_ph]
        self.buf_phs += values_as_sorted_list(pi_info_phs)

        # organize tf ops required for generation of actions
        self.ops_for_action = dict(pi=pi, 
                              v=v, 
                              logp_pi=logp_pi,
                              pi_info=pi_info)
        self.ops_for_action['vc'] = vc

        # Count variables
        var_counts = tuple(count_vars(scope) for scope in ['pi', 'vf', 'vc'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)

        # Make a sample estimate for entropy to use as sanity check
        approx_ent = tf.reduce_mean(-logp)

        # @anyboby borrowed from sac maybe for later ######
        target_entropy = kwargs['target_entropy']
        self._target_entropy = (
            -np.prod(self.act_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self.ent_reg = 0
        ##### ---------------- #####

        # ________________________________ #        
        #    Computation graph for policy  #
        # ________________________________ #
        ratio = tf.exp(logp-self.logp_old_ph)
        # Surrogate advantage / clipped surrogate advantage
        if self.agent.clipped_adv:
            min_adv = tf.where(self.adv_ph>0, 
                            (1+self.agent.clip_ratio)*self.adv_ph, 
                            (1-self.agent.clip_ratio)*self.adv_ph
                            )
            surr_adv = tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        else:
            surr_adv = tf.reduce_mean(ratio * self.adv_ph)

        # Surrogate cost (advantage)
        self.surr_cost = tf.reduce_mean(ratio * self.cadv_ph)

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

        elif self.agent.first_order:

            # Optimizer for first-order policy optimization
            train_pi = MpiAdamOptimizer(learning_rate= self.agent.pi_lr).minimize(self.pi_loss)

            # Prepare training package for agent
            self.training_package = dict(train_pi=train_pi)

        else:
            raise NotImplementedError

        # Provide training package to agent
        self.training_package.update(dict(pi_loss=self.pi_loss, 
                                    surr_cost=self.surr_cost,
                                    d_kl=self.d_kl, 
                                    target_kl=self.target_kl,
                                    cost_lim=self.cost_lim))
        self.agent.prepare_update(self.training_package)

        # ________________________________ #        
        #    Computation graph for value   #
        # ________________________________ #

        # Value losses
        self.v_loss = tf.reduce_mean((self.ret_ph - v)**2)
        self.vc_loss = tf.reduce_mean((self.cret_ph - vc)**2)

        # If agent uses penalty directly in reward function, don't train a separate
        # value function for predicting cost returns. (Only use one vf for r - p*c.)
        total_value_loss = self.v_loss + self.vc_loss

        # Optimizer for value learning
        self.train_vf = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(total_value_loss)

        # _____________________________________ #        
        #    Set up session, syncs and save     #
        # _____________________________________ #

        gpu = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
        )
        if gpu:
            ## --------------- allow dynamic memory growth to avoid cudnn init error ------------- ##
            from keras.backend.tensorflow_backend import set_session #---------------------------- ##
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,
                                        per_process_gpu_memory_fraction = 0.9/num_procs()), 
                                    log_device_placement=False)
            self.sess = tf.Session(config=config) #---------------------------------------------------- ##
            set_session(self.sess) # set this TensorFlow session as the default session for Keras ----- ##
            ## ----------------------------------------------------------------------------------- ##
        else:
            self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        
        # Sync params across processes
        self.sess.run(sync_all_params())

        # Setup model saving
        self.logger.setup_tf_saver(self.sess, inputs={'obs': self.obs_ph}, outputs={'pi': pi, 'v': v, 'vc': vc})

        # _____________________________________ #        
        #    Provide session to agent           #
        # _____________________________________ #
        self.agent.prepare_session(self.sess)

        #### -------------------------------- ####
        

    #@anyboby todo: buf_inputs has to be delivered to update. implement when buffer (or pool) is done
    def update(self, buf_inputs):
        cur_cost = self.logger.get_stats('EpCost')[0]
        rand_cost = 0
        cur_cost_lim = self.cost_lim-self._epoch*(self.cost_lim-self.cost_lim_end)/self._n_epochs + randint(0, rand_cost)
        c = cur_cost - cur_cost_lim
        if c > 0 and self.agent.cares_about_cost:
            self.logger.log('Warning! Safety constraint is already violated.', 'red')

        #=====================================================================#
        #  Prepare feed dict                                                  #
        #=====================================================================#

        inputs = {k:v for k,v in zip(self.buf_phs, buf_inputs)}     
        inputs[self.surr_cost_rescale_ph] = self.logger.get_stats('EpLen')[0]
        inputs[self.cur_cost_ph] = cur_cost

        #=====================================================================#
        #  Make some measurements before updating                             #
        #=====================================================================#

        measures = dict(LossPi=self.pi_loss,
                        SurrCost=self.surr_cost,
                        LossV=self.v_loss,
                        Entropy=self.ent)
        if not(self.agent.reward_penalized):
            measures['LossVC'] = self.vc_loss

        pre_update_measures = self.sess.run(measures, feed_dict=inputs)
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
        self.agent.update_pi(inputs)

        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#
        for _ in range(self.vf_iters):
            self.sess.run(self.train_vf, feed_dict=inputs)

        #=====================================================================#
        #  Make some measurements after updating                              #
        #=====================================================================#

        del measures['Entropy']
        measures['KL'] = self.d_kl

        post_update_measures = self.sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        self.logger.store(KL=post_update_measures['KL'], **deltas)


    def reset(self):
        pass

    def actions(self, obs):
        # check if single obs or multiple
        # remove single dims
        feed_obs = np.squeeze(obs)
        if len(feed_obs.shape) == len(self.obs_space.shape):
            feed_obs = feed_obs[np.newaxis]
        elif len(feed_obs.shape) > len(self.obs_space.shape):
            feed_obs = feed_obs
        else: 
            raise Exception('faulty obs')

        get_action_outs = self.sess.run(self.ops_for_action, 
                        feed_dict={self.obs_ph: feed_obs})
        a = get_action_outs['pi']
        return a


    def log_pis(self, obs, a):
        pass

    def get_action_outs(self, obs):
        get_action_outs = self.sess.run(self.ops_for_action, 
                        feed_dict={self.obs_ph: obs[np.newaxis]})
        return get_action_outs


    def actions_np(self, obs):
        actions = self.actions(obs)
        return np.array(actions)

    def log_pis_np(self, obs, a):
        pass
    def get_weights(self):
        raise NotImplementedError
        # return self.ac_network.get_weights

    def set_weights(self, *args, **kwargs):
        raise NotImplementedError
        # return self.ac_network.set_weights(*args, **kwargs)
        
    @property
    def trainable_variables(self):
        raise NotImplementedError
        # return self.ac_network.trainable_variables

    @property
    def non_trainable_weights(self):
        raise NotImplementedError
        # """Due to our nested model structure, we need to filter duplicates."""
        # return list(set(super(CPOPolicy, self).non_trainable_weights))

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        raise NotImplementedError

        (shifts_np,
        log_scale_diags_np,
        log_pis_np,
        raw_actions_np,
        actions_np) = self.diagnostics_model.predict(conditions)

        return OrderedDict({
            'shifts-mean': np.mean(shifts_np),
            'shifts-std': np.std(shifts_np),

            'log_scale_diags-mean': np.mean(log_scale_diags_np),
            'log_scale_diags-std': np.std(log_scale_diags_np),

            '-log-pis-mean': np.mean(-log_pis_np),
            '-log-pis-std': np.std(-log_pis_np),

            'raw-actions-mean': np.mean(raw_actions_np),
            'raw-actions-std': np.std(raw_actions_np),

            'actions-mean': np.mean(actions_np),
            'actions-std': np.std(actions_np),
        })
