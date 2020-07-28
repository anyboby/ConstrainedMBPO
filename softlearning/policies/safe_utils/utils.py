import numpy as np
import scipy.signal
import time 

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]

def discount_cumsum(x, discount, lam, weights=None, axis=0):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
        discount: factor for exponentially 
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    
    if n-D weights are provided, the output will look like this:
        [
            x0 * w0 + x1 * w1 * discount + x2 * w2 * discount^2,
            x1 * w0 + x2 * w1 * discount,
            x2 * w0
        ]
        
        for each element along the 0 axis of the inputs
    """

    if x.size>0:
        if weights is None:
            x_flipped = np.flip(x, axis=axis)
            disc_cumsum_flipped = scipy.signal.lfilter([1], [1, float(-discount*lam)], x_flipped, axis=axis)
            disc_cumsum = np.flip(disc_cumsum_flipped, axis=axis)
        else:
            # tic = time.time()
            # x_padded = np.concatenate((x, np.zeros_like(x)), axis=-1)             #### add zero padding to deltas to create matrix       
            # x_inds = np.arange(x.shape[-1])+np.arange(x.shape[-1])[...,None]   #### creates an index matrix with [[0,1,2,...][1,2,3,...][2,3,4,...]...]
            # x_mat = x_padded[...,x_inds]

            # seed = np.zeros(shape=x.shape[-1])
            # seed[0] = 1
            # discount_vec = scipy.signal.lfilter([1], [1, float(-discount)], seed)
            # discount_mat = np.repeat(discount_vec[None], repeats=x.shape[-1], axis=0)

            # weights_without_last = np.array(weights)
            # weights_without_last[...,-1] = 0
            # weights_padded = np.concatenate((weights_without_last, np.zeros_like(weights), np.zeros_like(weights)), axis=-1)  ### double padding, since l and t are indexed
            # w_inds = np.arange(weights.shape[-1]-1)+np.arange(weights.shape[-1])[...,None] + np.arange(weights.shape[-1])[...,None,None] ### indices, for choosing appropriate weights
            # lams = lam**np.repeat(w_inds[0][None], repeats=weights.shape[-1], axis=0)  #### lambdas don't increase with x-dim
            # final_lams = np.repeat(lam**(np.ones(shape=x.shape[-1])*x.shape[-1]-np.arange(1, x.shape[-1]+1))[...,None], repeats=x.shape[-1], axis=-1)
            # final_weights = np.ones(shape=x_mat.shape) * weights[...,-1][...,None,None] * final_lams
            # weights_lam = weights_padded[...,w_inds] * lams
            # weights_disc = discount_mat * ((1-lam) * np.sum(weights_lam, axis=-1) + final_weights) 
            # weights_disc_norm = weights_disc / weights_disc[...,0][...,None]
            
            # disc_cumsum = np.einsum('...ij, ...ji->...j', x_mat, weights_disc_norm)

            # toc = time.time()
            # print('old T:', toc-tic)
            # toc = time.time()

            w = np.array(weights)               ### copy weights
            lw = w[...,-1]                      ### last weight is handled seperately 
            lw = lw*(lam**w.shape[-1])              ## (accounts for whole projected weight after T)
            w[...,-1] = 0

            seed = np.zeros(shape=w.shape[-1])
            seed[0] = 1
            lam_vec = scipy.signal.lfilter([1], [1, float(-lam)], seed)     ### create vector of lambda-discounts
            
            w_fl = np.flip(w*lam_vec, axis=axis)                            ### combined weights of lambda * inv var
            w_lam = scipy.signal.lfilter([1], [1, float(-1)], w_fl, axis=axis) 
            w_lam = np.flip(w_lam, axis=axis)
            w_lam = (1-lam) * w_lam + lw[...,None]                         ### add back the final weight
            w_norm = np.array(w_lam)                                       ### normalization terms are equal to lambda weights at this point
            x_w = x*w_lam                                                  ### weight input vector
            x_w_fl = np.flip(x_w, axis=axis)                                
            x_w_fl_disc = scipy.signal.lfilter([1], [1, float(-discount)], x_w_fl, axis=axis)
            x_w_fl_disc = np.flip(x_w_fl_disc, axis=axis)
            disc_cumsum = x_w_fl_disc / w_norm                                   ### normalize result
    else: 
        disc_cumsum = np.array(x)
    return disc_cumsum
