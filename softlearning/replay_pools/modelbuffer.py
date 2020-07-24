import numpy as np
from softlearning.policies.safe_utils.mpi_tools import mpi_statistics_scalar
from softlearning.policies.safe_utils.utils import combined_shape, \
                             keys_as_sorted_list, \
                             values_as_sorted_list, \
                             discount_cumsum, \
                             EPS
from softlearning.replay_pools.cpobuffer import CPOBuffer

class ModelBuffer(CPOBuffer):

    def __init__(self, batch_size, env, max_path_length,
                 use_inv_var=False,
                 *args,
                 **kwargs,
                 ):
        
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.env = env
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        ## ignore other args and kwargs
        self.use_inv_var = use_inv_var

        obs_buf_shape = combined_shape(batch_size, combined_shape(max_path_length, self.obs_shape))
        act_buf_shape = combined_shape(batch_size, combined_shape(max_path_length, self.act_shape))
        scalar_shape = (batch_size, max_path_length)
        self.obs_buf = np.zeros(obs_buf_shape, dtype=np.float32)
        self.act_buf = np.zeros(act_buf_shape, dtype=np.float32)
        self.nextobs_buf = np.zeros(obs_buf_shape, dtype=np.float32)
        self.adv_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.adv_var_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.rew_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.rew_path_var_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.ret_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.val_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.val_var_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.cadv_buf = np.zeros(scalar_shape, dtype=np.float32)    # cost advantage
        self.cadv_var_buf = np.zeros(scalar_shape, dtype=np.float32)   
        self.cost_buf = np.zeros(scalar_shape, dtype=np.float32)    # costs
        self.cost_path_var_buf = np.zeros(scalar_shape, dtype=np.float32)    # costs
        self.cret_buf = np.zeros(scalar_shape, dtype=np.float32)    # cost return
        self.cval_buf = np.zeros(scalar_shape, dtype=np.float32)    # cost value
        self.cval_var_buf = np.zeros(scalar_shape, dtype=np.float32)    # cost value
        self.logp_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.term_buf = np.zeros(scalar_shape, dtype=np.bool_)
        
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

        self.ptr, self.path_start_idx, self.max_size, self.populated_mask, self.terminated_paths_mask = \
                                                                                0, \
                                                                                0, \
                                                                                np.ones(batch_size)*max_path_length, \
                                                                                np.zeros(scalar_shape, dtype=np.bool), \
                                                                                np.zeros(batch_size, dtype=np.bool)

    ''' initialize policy dependendant pi_info shapes, gamma, lam etc.'''
    def initialize(self, pi_info_shapes,
                    gamma=0.99, lam = 0.95,
                    cost_gamma = 0.99, cost_lam = 0.95,
                    ):
        self.pi_info_bufs = {k: np.zeros(shape=[self.batch_size]+[self.max_path_length] + list(v), dtype=np.float32) 
                            for k,v in pi_info_shapes.items()}
        self.sorted_pi_info_keys = keys_as_sorted_list(self.pi_info_bufs)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam

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

    def store_multiple(self, obs, act, next_obs, rew, rew_var, val, val_var, cost, cost_var, cval, cval_var, logp, pi_info, term):
        assertion_mask = self.ptr < self.max_size
        alive_paths = self.alive_paths
        assert assertion_mask.all()     # buffer has to have room so you can store
        assert len(obs)==alive_paths.sum()   # mismatch of alive paths and input obs ! call alive_paths !

        self.obs_buf[alive_paths, self.ptr] = obs
        self.act_buf[alive_paths, self.ptr] = act
        self.nextobs_buf[alive_paths, self.ptr] = next_obs
        self.rew_buf[alive_paths, self.ptr] = rew
        self.rew_path_var_buf[alive_paths, self.ptr] = rew_var
        self.val_buf[alive_paths, self.ptr] = val
        self.val_var_buf[alive_paths, self.ptr] = val_var
        self.cost_buf[alive_paths, self.ptr] = cost
        self.cost_path_var_buf[alive_paths, self.ptr] = cost_var
        self.cval_buf[alive_paths, self.ptr] = cval
        self.cval_var_buf[alive_paths, self.ptr] = cval_var
        self.logp_buf[alive_paths, self.ptr] = logp
        self.term_buf[alive_paths, self.ptr] = term
        for k in self.sorted_pi_info_keys:
            self.pi_info_bufs[k][alive_paths,self.ptr] = pi_info[k]
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
        
        if not self.use_inv_var:
            path_slice = slice(self.path_start_idx, self.ptr)
            
            rews = np.append(self.rew_buf[finish_mask, path_slice], last_val[..., np.newaxis], axis=1)
            vals = np.append(self.val_buf[finish_mask, path_slice], last_val[..., np.newaxis], axis=1)

            rew_vars = self.rew_path_var_buf[finish_mask, path_slice]-\
                        np.append(np.zeros(shape=(len(rews),1)), self.rew_path_var_buf[finish_mask, path_slice], axis=-1)[...,:-1]
            rew_vars = np.append(rew_vars, last_val_var[..., np.newaxis], axis=1)
            val_vars = np.append(self.val_var_buf[finish_mask, path_slice], last_val_var[..., np.newaxis], axis=1)
            
            #### inv var:
            deltas = rews[:,:-1] + self.gamma * vals[:, 1:] - vals[:, :-1]
            delta_vars = rew_vars[:,:-1] + self.gamma * val_vars[:, 1:] - val_vars[:, :-1]

            r_ret_vars = (self.rew_path_var_buf[finish_mask, path_slice] + self.val_var_buf[finish_mask, path_slice])
            r_vars_inv = 1/(r_ret_vars+EPS)
            
            self.adv_buf[finish_mask, path_slice] = discount_cumsum(deltas, self.gamma, self.lam, weights=r_vars_inv, axis=1)
            adv_vars = discount_cumsum(delta_vars, 1, self.lam, weights=r_vars_inv, axis=1) + self.val_var_buf[finish_mask, path_slice]
            self.adv_var_buf[finish_mask, path_slice] = adv_vars

            # deltas = rews[:,:-1] + self.gamma * vals[:, 1:] - vals[:, :-1]
            # self.adv_buf[finish_mask, path_slice] = discount_cumsum(deltas, self.gamma, self.lam, axis=1)

            # self.ret_buf[finish_mask, path_slice] = discount_cumsum(rews, self.gamma, axis=1)[:, :-1]
            self.ret_buf[finish_mask, path_slice] = self.adv_buf[finish_mask, path_slice] + self.val_buf[finish_mask, path_slice]

            costs = np.append(self.cost_buf[finish_mask, path_slice], last_cval[..., np.newaxis], axis=1)
            cvals = np.append(self.cval_buf[finish_mask, path_slice], last_cval[..., np.newaxis], axis=1)

            cost_vars = self.cost_path_var_buf[finish_mask, path_slice]-\
                            np.append(np.zeros(shape=(len(costs), 1)),self.cost_path_var_buf[finish_mask, path_slice], axis=-1)[...,:-1]
            cost_vars = np.append(cost_vars, last_cval_var[..., np.newaxis], axis=1)
            cval_vars = np.append(self.cval_var_buf[finish_mask, path_slice], last_cval_var[..., np.newaxis], axis=1)

            # self.cadv_buf[finish_mask, path_slice] = discount_cumsum(cdeltas, self.cost_gamma, self.cost_lam, axis=1)
            cdeltas = costs[:, :-1] + self.gamma * cvals[:, 1:] - cvals[:, :-1]
            cdelta_vars = cost_vars[:, :-1] + self.gamma * cval_vars[:, 1:] - cval_vars[:, :-1]
            
            cret_vars = (self.cost_path_var_buf[finish_mask, path_slice] + self.cval_var_buf[finish_mask, path_slice])
            c_vars_inv = 1/(cret_vars+EPS)

            self.cadv_buf[finish_mask, path_slice] = discount_cumsum(cdeltas, self.cost_gamma, self.cost_lam, weights=c_vars_inv, axis=1)
            cadv_vars = discount_cumsum(cdelta_vars, self.cost_gamma, self.cost_lam, weights=c_vars_inv, axis=1) + self.cval_var_buf[finish_mask, path_slice]
            self.cadv_var_buf[finish_mask, path_slice] = cadv_vars
            # self.cret_buf[finish_mask, path_slice] = discount_cumsum(costs, self.cost_gamma, axis=1)[:,:-1]
            self.cret_buf[finish_mask, path_slice] = self.cadv_buf[finish_mask, path_slice] + self.cval_buf[finish_mask, path_slice]

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
                self.cval_buf[self.populated_mask],
                self.cost_buf[self.populated_mask]] \
                + [v[self.populated_mask] for v in values_as_sorted_list(self.pi_info_bufs)]
        else:
            # Advantage normalizing trick for policy gradient
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[self.populated_mask].flatten())         # mpi can only handle 1d data
            self.adv_buf[self.populated_mask] = (self.adv_buf[self.populated_mask] - adv_mean) / (adv_std + EPS)

            # Center, but do NOT rescale advantages for cost gradient 
            # (since we're not just minimizing but aiming for a specific c)
            cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf[self.populated_mask].flatten())
            self.cadv_buf[self.populated_mask] -= cadv_mean

            res = [self.obs_buf, self.act_buf, self.adv_buf, 
                    self.cadv_buf, self.ret_buf, self.cret_buf, 
                    self.logp_buf, self.val_buf, self.cval_buf,
                    self.cost_buf] \
                    + values_as_sorted_list(self.pi_info_bufs)

            # filter out unpopulated entries / finished paths
            res = [buf[self.populated_mask] for buf in res]
            diagnostics = dict(returns_mean=res[4].mean(), creturns_mean = res[5].mean())

        # reset
        self.ptr, self.path_start_idx = 0, 0
        self.populated_mask = np.zeros(shape=self.populated_mask.shape, dtype=np.bool)
        self.terminated_paths_mask = np.zeros(shape=self.terminated_paths_mask.shape, dtype=np.bool)

        return res, diagnostics
