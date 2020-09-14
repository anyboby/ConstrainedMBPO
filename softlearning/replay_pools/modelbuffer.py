import numpy as np
from softlearning.policies.safe_utils.mpi_tools import mpi_statistics_scalar
from softlearning.policies.safe_utils.utils import *

from softlearning.replay_pools.cpobuffer import CPOBuffer
import scipy.signal

class ModelBuffer(CPOBuffer):

    def __init__(self, batch_size, env, max_path_length, ensemble_size,
                 use_inv_var=False,
                 *args,
                 **kwargs,
                 ):
        
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.env = env
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        self.use_inv_var = use_inv_var
        self.ensemble_size = ensemble_size
        self.model_ind = np.random.randint(ensemble_size)
        self.reset()
    

    ''' initialize policy dependendant pi_info shapes, gamma, lam etc.'''
    def initialize(self, pi_info_shapes,
                    gamma=0.99, lam = 0.95,
                    cost_gamma = 0.99, cost_lam = 0.95,
                    ):
        self.pi_info_bufs = {k: np.zeros(shape=[self.ensemble_size, self.batch_size]+[self.max_path_length] + list(v), dtype=np.float32) 
                            for k,v in pi_info_shapes.items()}
        self.sorted_pi_info_keys = keys_as_sorted_list(self.pi_info_bufs)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam

    def reset(self):
        obs_buf_shape = combined_shape(self.ensemble_size, combined_shape(self.batch_size, combined_shape(self.max_path_length, self.obs_shape)))
        act_buf_shape = combined_shape(self.ensemble_size, combined_shape(self.batch_size, combined_shape(self.max_path_length, self.act_shape)))
        ens_scalar_shape = (self.ensemble_size, self.batch_size, self.max_path_length)
        single_scalar_shape = (self.batch_size, self.max_path_length)

        self.obs_buf = np.zeros(obs_buf_shape, dtype=np.float32)
        self.act_buf = np.zeros(act_buf_shape, dtype=np.float32)
        self.nextobs_buf = np.zeros(obs_buf_shape, dtype=np.float32)
        self.adv_buf = np.zeros(single_scalar_shape, dtype=np.float32)
        self.ret_var_buf = np.zeros(single_scalar_shape, dtype=np.float32)
        self.ret_ep_var_buf = np.zeros(single_scalar_shape, dtype=np.float32)    ## epistemic value variance
        self.roll_lengths_buf = np.zeros(single_scalar_shape, dtype=np.float32)
        self.rew_buf = np.zeros(ens_scalar_shape, dtype=np.float32)
        self.rew_path_var_buf = np.zeros(ens_scalar_shape, dtype=np.float32)
        self.ret_buf = np.zeros(single_scalar_shape, dtype=np.float32)
        self.val_buf = np.zeros(ens_scalar_shape, dtype=np.float32)
        self.val_var_buf = np.zeros(ens_scalar_shape, dtype=np.float32)
        #self.val_ep_var_buf = np.zeros(ens_scalar_shape, dtype=np.float32)      ## epistemic value variance
        self.cadv_buf = np.zeros(single_scalar_shape, dtype=np.float32)    
        self.cret_var_buf = np.zeros(single_scalar_shape, dtype=np.float32)   
        self.cret_ep_var_buf = np.zeros(single_scalar_shape, dtype=np.float32)   ## epistemic cost return variance
        self.croll_lengths_buf = np.zeros(single_scalar_shape, dtype=np.float32)   
        self.cost_buf = np.zeros(ens_scalar_shape, dtype=np.float32)    
        self.cost_path_var_buf = np.zeros(ens_scalar_shape, dtype=np.float32)    
        self.cret_buf = np.zeros(single_scalar_shape, dtype=np.float32)    
        self.cval_buf = np.zeros(ens_scalar_shape, dtype=np.float32)    
        self.cval_var_buf = np.zeros(ens_scalar_shape, dtype=np.float32)    # cost value
        #self.cval_ep_var_buf = np.zeros(ens_scalar_shape, dtype=np.float32)    # epistemic cost value variance
        self.logp_buf = np.zeros(ens_scalar_shape, dtype=np.float32)
        self.term_buf = np.zeros(ens_scalar_shape, dtype=np.bool_)

        # ptr is a scalar to the current position in all paths. You are expected to store at the same timestep 
        #   in all parallel paths
        # path_start_idx is the path starting index, which will actually always be 0, since paths are parallel
        #   and always start at 0, may be removed 
        # max_size is actually also the same for all parallel paths, but a batch sized vector is more convenient
        #   for masked assertion
        # populated_mask shows us which entries in the buffer are valid, meaning they had a value stored in them
        #   and aren't terminated.
        # terminated_paths_mask essentially notes the same thing as populated_mask but is one_dimensional for 
        #   convenience

        self.ptr, self.path_start_idx, self.max_size, self.populated_mask, self.populated_indices, self.terminated_paths_mask = \
                                                            0, \
                                                            0, \
                                                            np.ones(self.batch_size)*self.max_path_length, \
                                                            np.zeros((self.batch_size, self.max_path_length), dtype=np.bool), \
                                                            np.repeat(np.arange(self.max_path_length)[None], axis=0, repeats=self.batch_size), \
                                                            np.zeros(self.batch_size, dtype=np.bool)


    @property
    def size(self):
        return self.populated_mask.sum()

    @property
    def has_room(self):
        room_mask = self.ptr < self.max_size

        return room_mask.all()

    @property
    def alive_paths(self):
        return np.logical_not(self.terminated_paths_mask)

    def store_multiple(self, obs, act, next_obs, rew, val, val_var, cost, cval, cval_var, logp, pi_info, term):
        assert (self.ptr < self.max_size).all()
        assert obs.shape[1]==self.alive_paths.sum()   # mismatch of alive paths and input obs ! call alive_paths !
        alive_paths = self.alive_paths
        
        self.obs_buf[:, alive_paths, self.ptr] = obs
        self.act_buf[:, alive_paths, self.ptr] = act
        self.nextobs_buf[:, alive_paths, self.ptr] = next_obs
        self.rew_buf[:, alive_paths, self.ptr] = rew
        #self.rew_path_var_buf[:, alive_paths, self.ptr] = rew_var
        self.val_buf[:, alive_paths, self.ptr] = val
        self.val_var_buf[:, alive_paths, self.ptr] = val_var
        self.cost_buf[:, alive_paths, self.ptr] = cost
        #self.cost_path_var_buf[:, alive_paths, self.ptr] = cost_var
        self.cval_buf[:, alive_paths, self.ptr] = cval
        self.cval_var_buf[:, alive_paths, self.ptr] = cval_var
        self.logp_buf[:, alive_paths, self.ptr] = logp
        self.term_buf[:, alive_paths, self.ptr] = term
        for k in self.sorted_pi_info_keys:
            self.pi_info_bufs[k][:, alive_paths, self.ptr] = pi_info[k]
        self.populated_mask[alive_paths, self.ptr] = True
        
        self.ptr += 1


    def finish_path_multiple(self, term_mask, last_val=0, last_val_var=0, last_cval=0, last_cval_var=0):
        """
        finishes multiple paths according to term_mask. 
        Note: if the term_mask indicates to terminate a path that has not yet been populated,
        it will terminate, but samples won't be marked as terminated (they won't be included 
        in get())
        Args:
            term_mask: a bool mask that indicates which paths should be terminated. 
                has to be of same length as currently alive paths.
            last_val: value of the last state in the paths that are to be finished.
                has to be of same length as the number of paths to be terminated (term_mask.sum())
            last_cval: cost value of the last state in the paths that are to be finished.
                has to be of same length as the number of paths to be terminated (term_mask.sum())
        """
        if not term_mask.any(): return                    ### skip if not terminating anything
        assert self.alive_paths.sum() == len(term_mask)   ### terminating a non-alive path!
        alive_paths = self.alive_paths

        ## concat masks for fancy indexing. (expand term_mask to buf dim)
        finish_mask = np.zeros(len(self.alive_paths), dtype=np.bool)
        finish_mask[tuple([alive[term_mask] for alive in np.where(alive_paths)])] = True
        
        if self.ptr>0:
            path_slice = slice(self.path_start_idx, self.ptr)
            
            rews = np.append(self.rew_buf[:, finish_mask, path_slice], last_val[..., None], axis=-1)
            vals = np.append(self.val_buf[:, finish_mask, path_slice], last_val[..., None], axis=-1)
            
            # @anyboby maybe remove GMM variance
            val_vars = np.append(self.val_var_buf[:, finish_mask, path_slice], last_val_var[..., None], axis=-1)
            val_vars = np.mean(val_vars, axis=0) + np.mean(vals**2, axis=0) - (np.mean(vals, axis=0))**2 

            #### only choose single trajectory for deltas
            deltas = rews[self.model_ind][...,:-1] + self.gamma * vals[self.model_ind][..., 1:] - vals[self.model_ind][..., :-1]

            #=====================================================================#
            #  Inverse Variance Weighted Advantages                               #
            #=====================================================================#
            
            ### define some utility indices
            t, t_p_h, h = triu_indices_t_h(size=deltas.shape[-1])
            Ht, HH = np.diag_indices(deltas.shape[-1])
            HH = np.flip(HH)    ### H is last value in each t-row

            ### define some utility vectors for lambda and gamma
            seed = np.zeros(shape=deltas.shape[-1]+1)
            seed[0] = 1
            disc_vec = scipy.signal.lfilter([1], [1, float(-self.gamma)], seed)     ### create vector of discounts
            disc_vec_sq = scipy.signal.lfilter([1], [1, float(-(self.gamma**2))], seed)     ### create vector of squared discounts
            lam_vec = scipy.signal.lfilter([1], [1, float(-self.lam*self.gamma**2)], seed)     ### create vector of lambdas 
                                                                                                ### (divide by gamma to get GAE for equal variances)

            ### calculate empirial epistemic variance per trajectory and rollout length
            ## @anyboby: for now without discount since we want the weighting to equal GAE for equal variances
            rew_t_h = disc_cumsum_matrix(self.rew_buf[:, finish_mask, path_slice], discount=self.gamma) #self.gamma)
            rew_t_h[..., t, h] += vals[..., t_p_h+1]*disc_vec[..., h+1]
            rew_var_t_h = np.var(rew_t_h, axis=0)       ### epistemic variances per timestep and rollout-length

            ### create inverse (epsitemic) variance matrix in t and rollout length h
            weight_mat = np.zeros_like(rew_var_t_h)
            weight_mat[...,t, h] = 1/(rew_var_t_h[..., t, h] + EPS) #* disc_vec[..., h+1]+EPS)

            ### add lambda weighting
            weight_mat[...,t, h] *= lam_vec[..., h]
            weight_mat[...,Ht, HH] *= 1/(1-self.lam*self.gamma**2+EPS)

            ### create weight matrix for deltas 
            d_weight_mat = discount_cumsum(weight_mat, 1.0, 1.0, axis=-1)               #### sum from l to H to get the delta-weight-matrix
            weight_norm = 1/d_weight_mat[..., 0]                                  #### normalize:
            d_weight_mat[...,t,h] = d_weight_mat[...,t,h]*weight_norm[..., t]     ####    first entry for every t containts sum of all weights
            
            ### calculate (epistemic) iv-weighted advantages
            self.adv_buf[finish_mask, path_slice] = discount_cumsum_weighted(deltas, self.gamma, d_weight_mat)
            ######################
            #### calculate aleatoric and epistemic return variances (later as variance correction targets)
            al_var_mat = rew_var_t_h.copy()
            al_var_mat[...,t,h] = rew_var_t_h[...,t,h] + val_vars[..., t_p_h+1]*disc_vec_sq[..., h+1]

            ### squared weights for variances
            al_var_weight_mat = al_var_mat.copy()
            al_var_weight_mat[...,t,h] = (weight_mat[...,t, h]*weight_norm[..., t])**2 * al_var_mat[...,t,h]
            
            ep_var_weight_mat = al_var_mat.copy()
            ep_var_weight_mat[...,t,h] = (weight_mat[...,t, h]*weight_norm[..., t])**2 * rew_var_t_h[...,t,h]

            self.ret_var_buf[finish_mask, path_slice] = \
                discount_cumsum_weighted(np.ones_like(deltas), 1.0, al_var_weight_mat)

            self.ret_ep_var_buf[finish_mask, path_slice] = \
                discount_cumsum_weighted(np.ones_like(deltas), 1.0, ep_var_weight_mat)

            self.roll_lengths_buf[finish_mask, path_slice] = \
                discount_cumsum_weighted(np.arange(self.ptr), 1.0, weight_mat)*weight_norm - np.arange(self.ptr)
            #### R_t = A_GAE,t^iv + V_t
            self.ret_buf[finish_mask, path_slice] = self.adv_buf[finish_mask, path_slice] + self.val_buf[self.model_ind, finish_mask, path_slice]

            # deltas = rews[:,:-1] + self.gamma * vals[:, 1:] - vals[:, :-1]
            # self.adv_buf[finish_mask, path_slice] = discount_cumsum(deltas, self.gamma, self.lam, axis=1)

            #=====================================================================#
            #  Inverse Variance Weighted Cost Advantages                          #
            #=====================================================================#
            
            ### define some utility vectors for lambda and gamma
            c_disc_vec = scipy.signal.lfilter([1], [1, float(-self.cost_gamma)], seed)     ### create vector of discounts
            c_disc_vec_sq = scipy.signal.lfilter([1], [1, float(-(self.cost_gamma**2))], seed)     ### create vector of squared discounts
            c_lam_vec = scipy.signal.lfilter([1], [1, float(-self.cost_lam*self.cost_gamma**2)], seed)     ### create vector of lambdas 

            costs = np.append(self.cost_buf[:, finish_mask, path_slice], last_cval[..., None], axis=-1)
            cvals = np.append(self.cval_buf[:, finish_mask, path_slice], last_cval[..., None], axis=-1)

            cval_vars = np.append(self.cval_var_buf[:, finish_mask, path_slice], last_cval_var[..., None], axis=-1)
            # @anyboby maybe remove GMM variance
            cval_vars = np.mean(cval_vars, axis=0) + np.mean(cval_vars**2, axis=0) - (np.mean(cval_vars, axis=0))**2 
            
            #### only choose single trajectory for deltas
            cdeltas = costs[self.model_ind][...,:-1] + self.cost_gamma * cvals[self.model_ind][..., 1:] - cvals[self.model_ind][..., :-1]

            ### calculate empirial epistemic variance per trajectory and rollout length
            ## @anyboby: for now without discount since we want the weighting to equal GAE for equal variances
            c_t_h = disc_cumsum_matrix(self.cost_buf[:, finish_mask, path_slice], discount=self.gamma) #self.gamma)
            c_t_h[..., t, h] += cvals[..., t_p_h+1]*c_disc_vec[..., h+1]
            c_var_t_h = np.var(c_t_h, axis=0)       ### epistemic variances per timestep and rollout-length
            
            ### create inverse (epsitemic) variance matrix in t and rollout length h
            c_weight_mat = np.zeros_like(c_var_t_h)
            c_weight_mat[...,t, h] = 1/(c_var_t_h[..., t, h] + EPS) #* disc_vec[..., h+1]+EPS)

            ### add lambda weighting
            c_weight_mat[...,t, h] *= c_lam_vec[..., h]
            c_weight_mat[...,Ht, HH] *= 1/(1-self.cost_lam*self.cost_gamma**2+EPS)

            ### create weight matrix for deltas 
            cd_weight_mat = discount_cumsum(c_weight_mat, 1.0, 1.0, axis=-1)               #### sum from l to H to get the delta-weight-matrix
            c_weight_norm = 1/cd_weight_mat[..., 0]                                  #### normalize:
            cd_weight_mat[...,t,h] = cd_weight_mat[...,t,h]*c_weight_norm[..., t]     ####    first entry for every t containts sum of all weights

            ### calculate (epistemic) iv-weighted advantages
            self.cadv_buf[finish_mask, path_slice] = discount_cumsum_weighted(cdeltas, self.cost_gamma, cd_weight_mat)
            ######################
            #### calculate aleatoric and epistemic return variances (later as variance correction targets)
            c_al_var_mat = c_var_t_h.copy()
            c_al_var_mat[...,t,h] = c_var_t_h[...,t,h] + cval_vars[..., t_p_h+1]*c_disc_vec_sq[..., h+1]

            ### squared weights for variances
            c_al_var_weight_mat = c_al_var_mat.copy()
            c_al_var_weight_mat[...,t,h] = (c_weight_mat[...,t, h]*c_weight_norm[..., t])**2 * c_al_var_mat[...,t,h]

            c_ep_var_weight_mat = c_al_var_mat.copy()
            c_ep_var_weight_mat[...,t,h] = (c_weight_mat[...,t, h]*c_weight_norm[..., t])**2 * c_var_t_h[...,t,h]

            self.cret_var_buf[finish_mask, path_slice] = \
                discount_cumsum_weighted(np.ones_like(cdeltas), 1.0, c_al_var_weight_mat)

            self.cret_ep_var_buf[finish_mask, path_slice] = \
                discount_cumsum_weighted(np.ones_like(deltas), 1.0, c_ep_var_weight_mat)

            self.croll_lengths_buf[finish_mask, path_slice] = \
                discount_cumsum_weighted(np.arange(self.ptr), 1.0, c_weight_mat)*c_weight_norm - np.arange(self.ptr)
            #### R_t = A_GAE,t^iv + V_t
            self.cret_buf[finish_mask, path_slice] = self.cadv_buf[finish_mask, path_slice] + self.cval_buf[self.model_ind, finish_mask, path_slice]
            
            #=====================================================================#
            #  Determine Rollout Lengths                                          #
            #=====================================================================#
            
            ### calc variance acceleration and normalize
            # c_var_acc = np.diff(np.diff(self.cret_var_buf[finish_mask, path_slice], axis=-1), axis=-1)
            # c_var_acc = (c_var_acc-np.mean(c_var_acc, axis=-1)[...,None])/(np.std(c_var_acc, axis=-1)[...,None]+EPS)
            # v_var_acc = np.diff(np.diff(self.ret_var_buf[finish_mask, path_slice], axis=-1), axis=-1)
            # v_var_acc = (v_var_acc-np.mean(v_var_acc, axis=-1)[...,None])/(np.std(v_var_acc, axis=-1)[...,None]+EPS)

            # var_acc_threshold = 1 ### all variance accelerations above 1 std are filtered out
            # var_acc_mask = np.logical_or(c_var_acc>var_acc_threshold, v_var_acc>var_acc_threshold)
            # horizons = np.argmax(var_acc_mask, axis=-1)[...,None] if var_acc_mask.size != 0 else np.zeros_like(finish_mask.sum())[...,None]
            # self.populated_mask[finish_mask,:] = self.populated_indices[finish_mask,:]<horizons

            ### alternative a: trajectory variance larger than epistemic value variance
            # val_var_threshold = 1.5
            # ep_cval_vars = np.var(self.cval_buf[:, finish_mask, path_slice], axis=0)
            # ep_val_vars = np.var(self.val_buf[:, finish_mask, path_slice], axis=0)

            # var_mask = np.logical_or(
            #     self.cret_var_buf[finish_mask, path_slice]>val_var_threshold * ep_cval_vars,
            #     self.ret_var_buf[finish_mask, path_slice]>val_var_threshold * ep_val_vars,
            #     )
            # horizons = np.argmax(var_mask, axis=-1)[...,None]
            # self.populated_mask[finish_mask,:] = self.populated_indices[finish_mask,:]<horizons

            ### alternative b: normalize return variances by first entry
            threshold = 5
            norm_cret_vars = self.cret_var_buf[finish_mask, path_slice]/(self.cret_var_buf[finish_mask, path_slice][...,0:1]+EPS)
            norm_ret_vars = self.ret_var_buf[finish_mask, path_slice]/(self.ret_var_buf[finish_mask, path_slice][...,0:1]+EPS)

            too_uncertain_mask = np.logical_or(
                norm_cret_vars>threshold,
                norm_ret_vars>threshold
            )
            horizons = np.argmax(too_uncertain_mask, axis=-1)[...,None]
            self.populated_mask[finish_mask,:] *= self.populated_indices[finish_mask,:]<horizons

        # mark terminated paths
        self.terminated_paths_mask += finish_mask
         
    def get(self):
        """
        Returns a list of predetermined values in the buffer.
        
        Returns:
            list: [self.obs_buf, self.act_buf, self.adv_buf,
                self.cadv_buf, self.ret_buf, self.cret_buf,
                self.logp_buf] + values_as_sorted_list(self.pi_info_bufs)
        """
        assert self.terminated_paths_mask.all()         ### all paths have to be finished

        if self.use_inv_var:
            #### _________________________________ ####
            ####      Inverse Variance Rollouts    ####
            #### _________________________________ ####
    
            start_states = self.obs_buf[self.populated_mask]
            returns, creturns, adv, c_adv, diagnostics = self.env.invVarRollout(
                        start_states,
                        gamma=self.gamma,
                        c_gamma=self.cost_gamma,
                        lam=1.03,            #self.lam
                        c_lam=1.04,          #self.cost_lam,
                        horizon=100,
                        stop_var=5e3
                    )
            res = [
                self.obs_buf[self.populated_mask], 
                self.act_buf[self.populated_mask], 
                adv, 
                c_adv, 
                returns, 
                creturns, 
                self.logp_buf[self.populated_mask], 
                self.val_buf[self.populated_mask], 
                self.val_var_buf[self.model_ind, self.populated_mask], 
                self.cval_buf[self.populated_mask],
                self.cval_var_buf[self.model_ind, self.populated_mask],
                self.cost_buf[self.populated_mask]] \
                + [v[self.populated_mask] for v in values_as_sorted_list(self.pi_info_bufs)]
        else:
            if self.size>0:
                # Advantage normalizing trick for policy gradient
                adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[self.populated_mask].flatten())         # mpi can only handle 1d data
                self.adv_buf[self.populated_mask] = (self.adv_buf[self.populated_mask] - adv_mean) / (adv_std + EPS)

                # Center, but do NOT rescale advantages for cost gradient 
                # (since we're not just minimizing but aiming for a specific c)
                cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf[self.populated_mask].flatten())
                self.cadv_buf[self.populated_mask] -= cadv_mean
                
                ret_mean = self.ret_buf[self.populated_mask].mean()
                cret_mean = self.cret_buf[self.populated_mask].mean()

                ep_val_var = self.val_var_buf[self.model_ind, self.populated_mask].mean()
                ep_cval_var = self.cval_var_buf[self.model_ind, self.populated_mask].mean()
                ep_val_var_mean = np.mean(np.var(self.val_buf[:, self.populated_mask], axis=0))
                ep_cval_var_mean = np.mean(np.var(self.cval_buf[:, self.populated_mask], axis=0))
                ep_ret_var_mean = np.mean(self.ret_ep_var_buf[self.populated_mask])
                ep_cret_var_mean = np.mean(self.cret_ep_var_buf[self.populated_mask])
                norm_adv_var_mean = np.mean(self.ret_var_buf[self.populated_mask])/np.var(self.adv_buf[self.populated_mask])
                norm_cadv_var_mean = np.mean(self.cret_var_buf[self.populated_mask])/np.var(self.cadv_buf[self.populated_mask])
                avg_horizon_r = np.mean(self.roll_lengths_buf[self.populated_mask])
                avg_horizon_c = np.mean(self.croll_lengths_buf[self.populated_mask])
            else:
                ret_mean = 0
                cret_mean = 0
                ep_val_var = 0
                ep_cval_var = 0
                ep_val_var_mean = 0
                ep_cval_var_mean = 0
                ep_ret_var_mean = 0
                ep_cret_var_mean = 0
                norm_adv_var_mean = 0
                norm_cadv_var_mean = 0
                avg_horizon_r = 0
                avg_horizon_c = 0


            res = [self.obs_buf[self.model_ind], self.act_buf[self.model_ind], self.adv_buf, self.ret_var_buf,
                    self.cadv_buf, self.cret_var_buf, self.ret_buf, self.cret_buf, 
                    self.logp_buf[self.model_ind], self.val_buf[self.model_ind], self.val_var_buf[self.model_ind], 
                    self.cval_buf[self.model_ind], self.cval_var_buf[self.model_ind], self.cost_buf[self.model_ind]] \
                    + [v[self.model_ind] for v in values_as_sorted_list(self.pi_info_bufs)]

            # filter out unpopulated entries / finished paths
            res = [buf[self.populated_mask] for buf in res]
            diagnostics = dict( poolm_batch_size = self.populated_mask.sum(), 
                                poolm_ret_mean=ret_mean, 
                                poolm_cret_mean=cret_mean, 
                                poolm_val_var_mean = ep_val_var,
                                poolm_cval_var_mean = ep_cval_var,
                                poolm_ep_val_var_mean = ep_val_var_mean,
                                poolm_ep_cval_var_mean = ep_cval_var_mean,
                                poolm_ep_ret_var_mean = ep_ret_var_mean,
                                poolm_ep_cret_var_mean = ep_cret_var_mean,
                                poolm_norm_adv_var = norm_adv_var_mean, 
                                poolm_norm_cadv_var = norm_cadv_var_mean,
                                poolm_avg_Horizon_rew = avg_horizon_r,
                                poolm_avg_Horizon_c = avg_horizon_c,
                                )
        # reset
        self.reset()
        # self.ptr, self.path_start_idx = 0, 0
        # self.populated_mask = np.zeros(shape=self.populated_mask.shape, dtype=np.bool)
        # self.terminated_paths_mask = np.zeros(shape=self.terminated_paths_mask.shape, dtype=np.bool)

        return res, diagnostics

