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

def discount_cumsum_weighted(x, lam, weights, axis=0):
    w = np.array(weights)               ### copy weights
    # lw = (w[...,-1] * lam**w.shape[-1] / (1-lam))[...,None]                      ### last weight is handled seperately 
    # w[...,-1] = 0              ## (accounts for whole projected weight after T)
    w[...,-1] = w[...,-1]/(1-lam)
    xw = x*w
    xw_fl = np.flip(xw, axis=-1)
    xw_fl = scipy.signal.lfilter([1], [1, float(-lam)], xw_fl, axis=axis)
    xw_lam = np.flip(xw_fl, axis=-1)
    norm_fl = scipy.signal.lfilter([1], [1, float(-lam)], np.flip(w, axis=-1), axis=axis)
    norm = np.flip(norm_fl, axis=-1)
    
    xw_norm = xw_lam/norm

    # seed = np.zeros(shape=w.shape[-1])
    # seed[0] = 1
    # lam_vec = scipy.signal.lfilter([1], [1, float(-lam)], seed)     ### create vector of lambda-discounts

    # w_lam = w*lam_vec
    # w_norm = scipy.signal.lfilter([1], [1, float(-1)], np.flip(w_lam, axis=-1), axis=axis)
    # w_norm = np.flip(w_norm, axis=-1)
    # w_norm = 1/(w_lam+lw)
    # w_lam_norm = w_lam*w_norm
    # xw = scipy.signal.lfilter([1], [1, -1.0], np.flip(x*w_lam, axis=-1), axis=axis)
    # xw = np.flip(xw,axis=-1)
    # xw_norm = (xw+lw*x[...,-1][...,None])*w_norm

    return xw_norm

