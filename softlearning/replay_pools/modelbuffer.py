import numpy as np
from softlearning.policies.safe_utils.mpi_tools import mpi_statistics_scalar
from softlearning.policies.safe_utils.utils import combined_shape, \
                             keys_as_sorted_list, \
                             values_as_sorted_list, \
                             discount_cumsum, \
                             EPS
from softlearning.replay_pools.cpobuffer import CPOBuffer

class ModelBuffer(CPOBuffer):

    def __init__(self, batch_size, max_path_length,
                 observation_space, action_space, 
                 *args,
                 **kwargs,
                 ):
        
        self.obs_shape = observation_space.shape
        self.act_shape = action_space.shape
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        ## ignore other args and kwargs

        obs_buf_shape = combined_shape(batch_size, combined_shape(max_path_length, self.obs_shape))
        act_buf_shape = combined_shape(batch_size, combined_shape(max_path_length, self.act_shape))
        scalar_shape = (batch_size, max_path_length)
        self.obs_buf = np.zeros(obs_buf_shape, dtype=np.float32)
        self.act_buf = np.zeros(act_buf_shape, dtype=np.float32)
        self.nextobs_buf = np.zeros(obs_buf_shape, dtype=np.float32)
        self.adv_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.rew_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.ret_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.val_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.cadv_buf = np.zeros(scalar_shape, dtype=np.float32)    # cost advantage
        self.cost_buf = np.zeros(scalar_shape, dtype=np.float32)    # costs
        self.cret_buf = np.zeros(scalar_shape, dtype=np.float32)    # cost return
        self.cval_buf = np.zeros(scalar_shape, dtype=np.float32)    # cost value
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

    def store_multiple(self, obs, act, next_obs, rew, val, cost, cval, logp, pi_info, term):
        assertion_mask = self.ptr < self.max_size
        not_term_mask = np.logical_not(self.terminated_paths_mask)
        assert assertion_mask.all()     # buffer has to have room so you can store
        self.obs_buf[not_term_mask, self.ptr] = obs[not_term_mask]
        self.act_buf[not_term_mask, self.ptr] = act[not_term_mask]
        self.nextobs_buf[not_term_mask, self.ptr] = next_obs[not_term_mask]
        self.rew_buf[not_term_mask, self.ptr] = rew[not_term_mask]
        self.val_buf[not_term_mask, self.ptr] = val[not_term_mask]
        self.cost_buf[not_term_mask, self.ptr] = cost[not_term_mask]
        self.cval_buf[not_term_mask, self.ptr] = cval[not_term_mask]
        self.logp_buf[not_term_mask, self.ptr] = logp[not_term_mask]
        self.term_buf[not_term_mask, self.ptr] = term[not_term_mask]
        for k in self.sorted_pi_info_keys:
            self.pi_info_bufs[k][not_term_mask,self.ptr] = pi_info[k][not_term_mask]
        self.populated_mask[not_term_mask, self.ptr] = True
        
        self.ptr += 1


    # def store_multiple(self, obs, act, rew, val, cost, cval, logp, pi_info, term):
    #     """
    #     stores samples of multiple paths running in parallel. Expects Args to be 
    #     numpy arrays. 
    #     """
    #     assert self.ptr < self.max_size     # buffer has to have room so you can store
    #     assert self.terminated_paths_mask.sum() < self.batch_size

    #     self.obs_buf[self.ptr] = obs
    #     self.act_buf[self.ptr] = act
    #     self.rew_buf[self.ptr] = rew
    #     self.val_buf[self.ptr] = val
    #     self.cost_buf[self.ptr] = cost
    #     self.cval_buf[self.ptr] = cval
    #     self.logp_buf[self.ptr] = logp
    #     self.term_buf[self.ptr] = term
    #     for k in self.sorted_pi_info_keys:
    #         self.pi_info_bufs[k][self.ptr] = pi_info[k]
    #     self.ptr += 1
    #     self.populated_mask[np.logical_not(self.terminated_paths_mask), self.ptr] = True

    def finish_path(self, last_val=0, last_cval=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        cdeltas = costs[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, self.cost_gamma * self.cost_lam)
        self.cret_buf[path_slice] = discount_cumsum(costs, self.cost_gamma)[:-1]

        self.path_start_idx = self.ptr

    def finish_path_multiple(self, term_mask, 
                                last_val=None, 
                                last_cval=None):
        assert term_mask.any()

        if last_cval is None:
            last_cval = np.zeros(self.batch_size)
        if last_val is None:
            last_val = np.zeros(self.batch_size)

        ### @anyboby TODO we're doing a lot of unnecessary calcs here, 
        # try to find a nicer way

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[:, path_slice], last_val[:, np.newaxis], axis=1)
        vals = np.append(self.val_buf[:, path_slice], last_val[:, np.newaxis], axis=1)
        deltas = rews[:,:-1] + self.gamma * vals[:, 1:] - vals[:, :-1]
        self.adv_buf[term_mask, path_slice] = discount_cumsum(deltas, self.gamma * self.lam, axis=1)[term_mask]
        self.ret_buf[term_mask, path_slice] = discount_cumsum(rews, self.gamma, axis=1)[term_mask, :-1]

        costs = np.append(self.cost_buf[:, path_slice], last_cval[:, np.newaxis], axis=1)
        cvals = np.append(self.cval_buf[:, path_slice], last_cval[:, np.newaxis], axis=1)
        cdeltas = costs[:, :-1] + self.gamma * cvals[:, 1:] - cvals[:, :-1]
        self.cadv_buf[term_mask, path_slice] = discount_cumsum(cdeltas, self.cost_gamma * self.cost_lam, axis=1)[term_mask]
        self.cret_buf[term_mask, path_slice] = discount_cumsum(costs, self.cost_gamma, axis=1)[term_mask,:-1]

        # mark terminated paths
        self.terminated_paths_mask += term_mask

    def get(self):
        """
        Returns a list of predetermined values in the buffer.
        
        Returns:
            list: [self.obs_buf, self.act_buf, self.adv_buf,
                self.cadv_buf, self.ret_buf, self.cret_buf,
                self.logp_buf] + values_as_sorted_list(self.pi_info_bufs)
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
                                                # This doesn't make sense for parallel paths:
                                                # buffer doesn't have to be full to get
        
        # Advantage normalizing trick for policy gradient
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[self.populated_mask].flatten())         # mpi can only handle 1d data
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + EPS)

        # Center, but do NOT rescale advantages for cost gradient 
        # (since we're not just minimizing but aiming for a specific c)
        cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf[self.populated_mask].flatten())
        self.cadv_buf -= cadv_mean

        res = [self.obs_buf, self.act_buf, self.adv_buf, 
                self.cadv_buf, self.ret_buf, self.cret_buf, 
                self.logp_buf, self.val_buf, self.cval_buf] \
                + values_as_sorted_list(self.pi_info_bufs)

        # filter out unpopulated entries / finished paths
        res = [buf[self.populated_mask] for buf in res]

        # reset
        self.ptr, self.path_start_idx = 0, 0
        self.populated_mask = np.zeros(shape=self.populated_mask.shape, dtype=np.bool)
        self.terminated_paths_mask = np.zeros(shape=self.terminated_paths_mask.shape, dtype=np.bool)

        return res

    # def get_model_samples(self):
    #     # buffer does not have to be full for model samples
    #     # self.ptr, self.path_start_idx = 0, 0

    #     # Advantage normalizing trick for policy gradient
    #     adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
    #     self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + EPS)

    #     # Center, but do NOT rescale advantages for cost gradient
    #     cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf)
    #     self.cadv_buf -= cadv_mean

    #     obs = self.obs_buf[:self.ptr-1]
    #     next_obs = self.obs_buf[1:self.ptr]
    #     acts = self.act_buf[:self.ptr-1]
    #     rews = self.rew_buf[:self.ptr-1, np.newaxis]
    #     costs = self.cost_buf[:self.ptr-1, np.newaxis]
    #     terms = self.term_buf[:self.ptr-1, np.newaxis]
    #     samples = {
    #         'observations':obs,
    #         'next_observations':next_obs,
    #         'actions':acts,
    #         'rewards':rews,
    #         'costs':costs,
    #         'terminals':terms,
    #     }
    #     return samples