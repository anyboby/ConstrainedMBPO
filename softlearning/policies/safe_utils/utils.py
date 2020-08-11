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

def discount_cumsum_weighted(x, disc, weights, axis=-1):
    '''
    Calculates a discounted cumulated sum with weights. weights should be a matrix, i.e.
    have one more dimension than x, s.t.:
    dcw_t = \sum_l^{T-t}(  {lam^l * x_{t+l} * w_{t,l}  )
    Args: 
        x: the ndarray
        lam: scalar for discounting
        weights: weight matrix with dim = dim(x)+1 and same lengths as x along all dims
    '''

    #### expand x
    #### essentially repeats the array along given axis and appends after the given axis
    # x_exp = np.repeat(x[...,None], repeats=x.shape[-1], axis=-1)      

    #### create discount vector
    seed = np.zeros(shape=x.shape[-1])
    seed[0] = 1
    disc_vec = scipy.signal.lfilter([1], [1, float(-disc)], seed)     ### create vector of discounts

    t, t_p_h, h = triu_indices_t_h(x.shape[-1])

    res = np.zeros_like(weights)
    res[...,t, h] = disc_vec[..., h] * x[...,t_p_h] * weights[...,t,h]
    res = np.add.reduce(res, axis=-1)

    return res

    # w = np.array(weights)               ### copy weights
    # # lw = (w[...,-1] * lam**w.shape[-1] / (1-lam))[...,None]                      ### last weight is handled seperately 
    # # w[...,-1] = 0              ## (accounts for whole projected weight after T)
    # w[...,-1] = w[...,-1]/(1-lam)
    # xw = x*w
    # xw_fl = np.flip(xw, axis=-1)
    # xw_fl = scipy.signal.lfilter([1], [1, float(-lam)], xw_fl, axis=axis)
    # xw_lam = np.flip(xw_fl, axis=-1)
    # norm_fl = scipy.signal.lfilter([1], [1, float(-lam)], np.flip(w, axis=-1), axis=axis)
    # norm = np.flip(norm_fl, axis=-1)
    
    # xw_norm = xw_lam/norm

    # # seed = np.zeros(shape=w.shape[-1])
    # # seed[0] = 1
    # # lam_vec = scipy.signal.lfilter([1], [1, float(-lam)], seed)     ### create vector of lambda-discounts

    # # w_lam = w*lam_vec
    # # w_norm = scipy.signal.lfilter([1], [1, float(-1)], np.flip(w_lam, axis=-1), axis=axis)
    # # w_norm = np.flip(w_norm, axis=-1)
    # # w_norm = 1/(w_lam+lw)
    # # w_lam_norm = w_lam*w_norm
    # # xw = scipy.signal.lfilter([1], [1, -1.0], np.flip(x*w_lam, axis=-1), axis=axis)
    # # xw = np.flip(xw,axis=-1)
    # # xw_norm = (xw+lw*x[...,-1][...,None])*w_norm

    # return xw_norm

def disc_cumsum_matrix (x, discount, max_size=100):
    '''
    creates a matrix that contains discounted sums of subarrays of the given array x. 
    the result is 1-more dimensional than the provided array, due to adding 
    the index t over the original array over h (the horizon of the sub-sum)
    for example:
        x = [a,b,c]
        discount = .5
        result = [[a, a+.5b, a+.5b+.25c],
                  [b, b+.5c, 0         ],
                  [c, 0    , 0         ]]
    
    Args:
        x: the nd array
        discount: discount scalar as described above
        max_size: if provided, the resulting matrix is shrunk to max_size x max_size, 
                    where the entries are simply added. For example:
                        x = [a,b,c,d]
                        discount= .5
                        max_size=2
                        result = [[a+b, a+b + .25(c+d)],
                                  [c+d, 0            ]]
    '''

    T = x.shape[-1]
    #### reducing somehow only accepts exlusive final indices
    x_pad = np.append(x, np.zeros(shape=x.shape[:-1])[...,None], axis=-1)
    
    #### contract array to max size
    size = min(max_size+1, T+1)
    contraction_ratio = (T+1 )/ size
    segment_inds = np.linspace(0,T, size, dtype=np.int)
    segments = np.add.reduceat(x_pad, segment_inds, axis=-1)

    #### get indices for t and H
    t, t_p_H, H = triu_indices_t_h(size-1)
    reduce_inds = np.ravel([t,t_p_H+1], 'F')        #### produce indices s.t. we have t,t+1,t,t+2,t,t+3... for reduceat
    
    #### create discount amtrix
    seed = np.zeros(shape=size)
    seed[0] = 1
    #surr_disc = (1-discount**contraction_ratio) / (contraction_ratio*(1-discount))  ### calc surrogate discount for average of e.g. (1+.99+.99^2+.99^3)/4
    disc_vec = scipy.signal.lfilter([1], [1, float(-discount**contraction_ratio)], seed)     ### create vector of discounts (discounts along H)
    #disc_mat = np.repeat(disc_vec[None], repeats=T, axis=0)               ### repeat discs for each entry along t

    #### reduce x segments
    x_reduced = np.add.reduceat(x_pad*disc_vec, reduce_inds, axis=-1)[...,::2]
    x_reduced /= disc_vec[t]

    x_mat = np.zeros((x.shape[:-1]+(size-1, size-1)))
    #x_reduced_mat = np.zeros(x.shape[:-1]+(size-1, size-1))
    x_mat[...,t,H] = x_reduced

    return x_mat
    
def triu_indices_t_h (size):
    t = np.triu_indices(size)[0]
    t_p_H = np.triu_indices(size)[1]
    H = t_p_H-t

    return t, t_p_H, H