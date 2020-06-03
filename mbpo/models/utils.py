from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
EPS = 1e-10

def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val

def gaussian_kl_np(mu0, log_std0, mu1, log_std1):
    """interprets each entry in mu_i and log_std_i as independent, 
    preserves shape
    output clipped to {0, 1e10}
    """
    var0, var1 = np.exp(2 * log_std0), np.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1- mu0)**2 + var0)/(var1+EPS) - 1) +  log_std1 - log_std0
    all_kls = pre_sum
    #all_kls = np.mean(all_kls)
    all_kls = np.clip(all_kls, 0, 1/EPS)        ### for stability
    return all_kls
def gaussian_jsd_np(mu0, log_std0, mu1, log_std1):
    pass
def average_dkl(mu, std):
    """
    Calculates the average kullback leiber divergences of multiple  univariate gaussian distributions.
    
    K(P1,…Pk) = 1/(k(k−1)) ∑_[k_(i,j)=1] DKL(Pi||Pj)
    
        (Andrea Sgarro, Informational divergence and the dissimilarity of probability distributions.)
    
    expects the distributions along axis 0, and samples along axis 1.
    Output is reduced by axis 0

    Args:
        mu: array-like means
        std: array-like stds
    """
    ## clip log
    log_std = np.log(std)
    log_std = np.clip(log_std, -100, 1e8)
    assert len(mu.shape)>=2 and len(log_std.shape)>=2
    num_models = len(mu)
    d_kl = None
    for i in range(num_models):
        for j in range(num_models):
            if d_kl is None:
                d_kl = gaussian_kl_np(mu[i], log_std[i], mu[j], log_std[j])
            else: d_kl+= gaussian_kl_np(mu[i], log_std[i], mu[j], log_std[j])
    d_kl = d_kl/(num_models*(num_models-1)+EPS)
    return d_kl



class TensorStandardScaler:
    """Helper class for automatically normalizing inputs into the network.
    """
    def __init__(self, x_dim, sc_factor=1):
        """Initializes a scaler.

        Arguments:
        x_dim (int): The dimensionality of the inputs into the scaler.

        Returns: None.
        """
        self.fitted = False
        with tf.variable_scope("Scaler"):
            self.mu = tf.get_variable(
                name="scaler_mu", shape=[1, x_dim], initializer=tf.constant_initializer(0.0),
                trainable=False
            )
            self.sigma = tf.get_variable(
                name="scaler_std", shape=[1, x_dim], initializer=tf.constant_initializer(1.0),
                trainable=False
            )

        self.cached_mu, self.cached_sigma = np.zeros([0, x_dim]), np.ones([1, x_dim])
        self.sc_factor = sc_factor

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        self.mu.load(mu)
        self.sigma.load(sigma)
        self.fitted = True
        self.cache()

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.
        
        can be adjusted to scale with a factor, to control sensitivity to ood data:
        d = (d-mu)/sigma = d + (d-mu)/sigma - d = d + (d(1-sigma)-mu)/sigma 
        and the version with scaling factor thus becomes
        d = d + sc_factor*(d(1-sigma)-mu)/sigma

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        sc_factor: Factor to what degree the original dataset is transformed

        Returns: (np.array) The transformed dataset.


        """
        #scaled_transform = data + self.sc_factor * (data* (1-self.sigma) - self.mu) / self.sigma
        scaling = 1+self.sc_factor*(self.sigma-1)
        scaling = tf.clip_by_value(scaling, 1.0e-8, 1.0e8)
        scaled_transform = (data-self.mu)/scaling
        return scaled_transform

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.sigma * data + self.mu

    def get_vars(self):
        """Returns a list of variables managed by this object.

        Returns: (list<tf.Variable>) The list of variables.
        """
        return [self.mu, self.sigma]

    def cache(self):
        """Caches current values of this scaler.

        Returns: None.
        """
        self.cached_mu = self.mu.eval()
        self.cached_sigma = self.sigma.eval()

    def load_cache(self):
        """Loads values from the cache

        Returns: None.
        """
        self.mu.load(self.cached_mu)
        self.sigma.load(self.cached_sigma)

