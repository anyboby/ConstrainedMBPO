import numpy as np
from softlearning.policies.safe_utils.mpi_tools import mpi_statistics_scalar
from softlearning.policies.safe_utils.utils import combined_shape, \
                             keys_as_sorted_list, \
                             values_as_sorted_list, \
                             discount_cumsum, \
                             EPS

import warnings
import random

class CPOBuffer:

    def __init__(self, size, archive_size, 
                 observation_space, action_space, 
                 *args,
                 **kwargs,
                 ):
        self.obs_shape = observation_space.shape
        self.act_shape = action_space.shape
        ## ignore other args and kwargs

        # _____________________________ #
        # Create buffers and archives   #
        # _____________________________ #

        #### The buffers are for on-policy learning and get erased with 
        # every call to get(). They are then copied to an archive where they remain 
        # accessable for model learning or off-policy learning. 
        # 
        # @anyboby: TODO some entries, like the 
        # policy dependent value f are probably wasted memory in the archive        

        self.obs_buf = np.zeros(combined_shape(size, self.obs_shape), dtype=np.float32)
        self.obs_archive = np.zeros(combined_shape(archive_size, self.obs_shape), dtype=np.float32)

        self.act_buf = np.zeros(combined_shape(size, self.act_shape), dtype=np.float32)
        self.act_archive = np.zeros(combined_shape(archive_size, self.act_shape), dtype=np.float32)

        # bit memory inefficient, but more convenient, to store next_obs
        self.nextobs_buf = np.zeros(combined_shape(size, self.obs_shape), dtype=np.float32)          
        self.nextobs_archive = np.zeros(combined_shape(archive_size, self.obs_shape), dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.adv_archive = np.zeros(archive_size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.rew_archive = np.zeros(archive_size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.ret_archive = np.zeros(archive_size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.val_archive = np.zeros(archive_size, dtype=np.float32)

        self.cadv_buf = np.zeros(size, dtype=np.float32)    # cost advantage
        self.cadv_archive = np.zeros(archive_size, dtype=np.float32)    # cost advantage

        self.cost_buf = np.zeros(size, dtype=np.float32)    # costs
        self.cost_archive = np.zeros(archive_size, dtype=np.float32)    # costs

        self.cret_buf = np.zeros(size, dtype=np.float32)    # cost return
        self.cret_archive = np.zeros(archive_size, dtype=np.float32)    # cost return

        self.cval_buf = np.zeros(size, dtype=np.float32)    # cost value
        self.cval_archive = np.zeros(archive_size, dtype=np.float32)    # cost value

        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.logp_archive = np.zeros(archive_size, dtype=np.float32)

        self.term_buf = np.zeros(size, dtype=np.bool_)
        self.term_archive = np.zeros(archive_size, dtype=np.bool_)
        
        self.buf_dict = {
            'observations': self.obs_buf,
            'actions':      self.act_buf,
            'next_observations': self.nextobs_buf,
            'advantages':   self.adv_buf,
            'rewards':      self.rew_buf,
            'returns':      self.ret_buf,
            'values':       self.val_buf,
            'cadvantages':  self.cadv_buf,
            'costs':        self.cost_buf,
            'creturns':     self.cret_buf,
            'cvalues':      self.cval_buf,
            'log_policies': self.logp_buf,
            'terminals':    self.term_buf,
        }

        self.arch_dict = {
            'observations': self.obs_archive,
            'actions':      self.act_archive,
            'next_observations': self.nextobs_archive,
            'advantages':   self.adv_archive,
            'rewards':      self.rew_archive,
            'returns':      self.ret_archive,
            'values':       self.val_archive,
            'cadvantages':  self.cadv_archive,
            'costs':        self.cost_archive,
            'creturns':     self.cret_archive,
            'cvalues':      self.cval_archive,
            'log_policies': self.logp_archive,
            'terminals':    self.term_archive,
        }

        self.ptr, self.path_start_idx, self.archive_ptr, self.path_finished = 0,0,0, False
        self.max_size, self.archive_size, self.archive_full = size, archive_size, False

    ''' initialize policy dependendant pi_info shapes, gamma, lam etc.'''
    def initialize(self, pi_info_shapes,
                    gamma=0.99, lam = 0.95,
                    cost_gamma = 0.99, cost_lam = 0.95,
                    ):
        self.pi_info_bufs = {k: np.zeros([self.max_size] + list(v), dtype=np.float32) 
                            for k,v in pi_info_shapes.items()}
        self.pi_info_archive = {k: np.zeros([self.archive_size] + list(v), dtype=np.float32) 
                            for k,v in pi_info_shapes.items()}

        self.sorted_pi_info_keys = keys_as_sorted_list(self.pi_info_bufs)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam

    @property
    def size(self):
        return self.ptr

    @property
    def arch_size(self):
        if not self.archive_full:
            return self.archive_ptr
        else: return self.archive_size

    def store(self, obs, act, next_obs, rew, val, cost, cval, logp, pi_info, term):
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.nextobs_buf[self.ptr] = next_obs
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.cost_buf[self.ptr] = cost
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp
        self.term_buf[self.ptr] = term
        for k in self.sorted_pi_info_keys:
            self.pi_info_bufs[k][self.ptr] = pi_info[k]
        self.ptr += 1
        self.path_finished = False

    def finish_path(self, last_val=0, last_cval=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]

        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        cdeltas = costs[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, self.cost_gamma * self.cost_lam)
        
        # testing lambda - TD
        # self.cret_buf[path_slice] = discount_cumsum(costs, self.cost_gamma)[:-1]
        self.cret_buf[path_slice] = self.cadv_buf[path_slice] + self.cval_buf[path_slice]
        
        self.path_start_idx = self.ptr
        self.path_finished = True

    def dump_to_archive(self):
        """
        dumps all samples that are currently contained in the buffers to 
        the archive. Careful: use this only, when all paths are finished, 
        otherwise the copied data is incomplete

        resets pointers to zero.

        """
        assert self.path_finished
        
        if self.archive_ptr >= self.archive_size-self.ptr:
            self.archive_full = True
            self.archive_ptr=0
            warnings.warn('Archive is full, deleting old samples.')

        arch_slice = slice(self.archive_ptr, self.archive_ptr+self.ptr)
        buf_slice = slice(0, self.ptr)

        self.obs_archive[arch_slice] = self.obs_buf[buf_slice]
        self.act_archive[arch_slice] = self.act_buf[buf_slice]
        self.nextobs_archive[arch_slice] = self.nextobs_buf[buf_slice]
        self.rew_archive[arch_slice] = self.rew_buf[buf_slice]
        self.val_archive[arch_slice] = self.val_buf[buf_slice]
        self.cost_archive[arch_slice] = self.cost_buf[buf_slice]
        self.cval_archive[arch_slice] = self.cval_buf[buf_slice]
        self.logp_archive[arch_slice] = self.logp_buf[buf_slice]
        self.term_archive[arch_slice] = self.term_buf[buf_slice]
        self.adv_archive[arch_slice] = self.adv_buf[buf_slice]
        self.ret_archive[arch_slice] = self.ret_buf[buf_slice]
        self.cadv_archive[arch_slice] = self.cadv_buf[buf_slice]
        self.cret_archive[arch_slice] = self.cret_buf[buf_slice]

        for k in self.sorted_pi_info_keys:
            self.pi_info_archive[k][arch_slice] = self.pi_info_bufs[k][buf_slice]
        
        self.archive_ptr+=self.ptr
        self.ptr, self.path_start_idx = 0,0
        
    def get(self):
        """
        Returns a list of predetermined values in the buffer.
        
        Returns:
            list: [self.obs_buf, self.act_buf, self.adv_buf,
                self.cadv_buf, self.ret_buf, self.cret_buf,
                self.logp_buf] + values_as_sorted_list(self.pi_info_bufs)
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.dump_to_archive()  # resets pointers

        # Advantage normalizing trick for policy gradient
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + EPS)

        # Center, but do NOT rescale advantages for cost gradient
        cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf)
        self.cadv_buf -= cadv_mean

        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.cadv_buf, self.ret_buf, self.cret_buf,
                self.logp_buf, self.val_buf, self.cval_buf,
                self.cost_buf] \
                + values_as_sorted_list(self.pi_info_bufs)
    
    def get_archive(self, fields=None):
        """
        returns all data currently contained in the buffer. if fields is None a 
        default dictionary containing:
        obs, action, next_obs, r, term
        is returned. 
        Remember, that old values / advantages etc. might come from
        old policies.
        choose from these fields:
            'observations'
            'actions'
            'next_observations'
            'advantages'
            'rewards'
            'returns'
            'values'
            'cadvantages'
            'costs'
            'creturns'
            'cvalues'
            'log_policies'
            'terminals'

        Args:
            fields: a list containing the key words for the desired 
            data types. e.g.: ['observations', 'actions', 'values']
        """
        if fields is None:
            archives = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
        else:
            archives = fields

        if self.archive_full:
            arch_ptr = self.archive_size
        else:
            arch_ptr = self.archive_ptr
        buf_ptr = self.ptr

        samples = {archive: \
                        np.concatenate((self.arch_dict[archive][:arch_ptr], self.buf_dict[archive][:buf_ptr]), axis=0) \
                        for archive in archives}
        # add current buffers
        return samples

    def rand_batch_from_archive(self, batch_size, fields=None):
        """
        returns random batch from the archive. if data_type is None a 
        default dictionary containing:
        obs, action, next_obs, r, term
        is returned. 
        Remember, that old values / advantages etc. might come from
        old policies.
        choose from these data types:
            'observations'
            'actions'
            'next_observations'
            'advantages'
            'rewards'
            'returns'
            'values'
            'cadvantages'
            'costs'
            'creturns'
            'cvalues'
            'log_policies'
            'terminals'

        Args:
            fields: a list containing the key words for the desired 
            data types. e.g.: ['observations', 'actions', 'values']
        Returns:
            samples: A dict containining random archive values for default / specified fields 
            as np arrays of batch_size
        """
        if fields is None:
            archives = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
        else:
            archives = fields

        rand_indc = self._random_indices(batch_size)
        samples = {archive: self.arch_dict[archive][rand_indc] for archive in archives}
        
        return samples

    def _random_indices(self, batch_size):
        """
        returns np array of random indices limited to current archive size.
        """

        if self.arch_size == 0: return np.arange(0, 0)
        #return random.sample(range(0, self.arch_size), batch_size)
        return np.random.randint(0, self.arch_size, batch_size)


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