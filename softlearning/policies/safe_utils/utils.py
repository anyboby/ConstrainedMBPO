import numpy as np
import scipy.signal

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]

def discount_cumsum(x, discount, axis=0):
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
    """

    x_flipped = np.flip(x, axis=axis)
    disc_cumsum_flipped = scipy.signal.lfilter([1], [1, float(-discount)], x_flipped, axis=axis)
    disc_cumsum = np.flip(disc_cumsum_flipped, axis=axis)

    # original 
    #scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    return disc_cumsum
