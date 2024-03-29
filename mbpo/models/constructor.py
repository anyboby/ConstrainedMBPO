import numpy as np
import numpy.ma as ma
import tensorflow as tf

import copy
from mbpo.models.fc import FC
from mbpo.models.bnn import BNN

def construct_model(in_dim, 
					out_dim,
					name='BNN',
					hidden_dims=(200, 200, 200), 
					num_networks=7, 
					num_elites=5,
					loss = 'NLL', 
					activation = 'swish',
					output_activation = None,
					decay=1e-4,
					lr = 1e-3,
					lr_decay = None,
					decay_steps=None,
					weighted=False, 
					use_scaler_in = False,
					use_scaler_out = False,
					sc_factor = 1,
					clip_loss = False,
					kl_cliprange = 0.1,
					var_corr = False,
					max_logvar = .5,
					min_logvar = -6,
					logit_bias_std=1.0,
					session=None):
	"""
	Constructs a tf model.
	Args:
		loss: Choose from 'NLL', 'MSE', 'Huber', or 'CE'. 
				choosing NLL will construct a model with variance output
	"""
	print('[ BNN ] dim in / out: {} / {} | Hidden dim: {}'.format(in_dim, out_dim, hidden_dims))
	#print('[ BNN ] Input Layer dim: {} | Output Layer dim: {} '.format(obs_dim_in+act_dim+prior_dim, obs_dim_out+rew_dim))
	params = {'name': name, 
				'loss':loss, 
				'num_networks': num_networks, 
				'num_elites': num_elites, 
				'sess': session,
				'use_scaler_in': use_scaler_in,
				'use_scaler_out': use_scaler_out,
				'sc_factor': sc_factor,
				'clip_loss': clip_loss,
				'kl_cliprange':kl_cliprange,
				'var_corr': var_corr,
				'max_logvar':max_logvar,
				'min_logvar':min_logvar,
				'logit_bias_std':logit_bias_std,
				}
	model = BNN(params)
	model.add(FC(hidden_dims[0], input_dim=in_dim, activation=activation, weight_decay=decay/4))	# def dec: 0.000025))
	
	for hidden_dim in hidden_dims[1:]:
		model.add(FC(hidden_dim, activation=activation, weight_decay=decay/2))						# def dec: 0.00005))
	
	model.add(FC(out_dim, activation=output_activation, weight_decay=decay))						# def dec: 0.0001
	
	opt_params = {"learning_rate":lr} if lr_decay is None else {"learning_rate":lr, 
																"learning_rate_decay":lr_decay,
																"decay_steps":decay_steps}
	model.finalize(tf.train.AdamOptimizer, opt_params, weighted=weighted)

	total_parameters = 0
	for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name):
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	print('[ BNN ] Total trainable Parameteres: {} '.format(total_parameters))


	return model

def format_samples_for_dyn(samples, priors = None, noise=None):
	"""
	formats samples to fit training, specifically returns: 

	if priors are given, they will be concatenated to the inputs. 
	priors have to be the same length in the first dimension as the samples
	given (that is, the number of samples)

	inputs, outputs:

	inputs = np.concatenate((observations, act, priors), axis=-1)
	outputs = np.concatenate((delta_observations, rewards (,costs)), axis=-1)

	"""
	obs = np.squeeze(samples['observations'])
	act = np.squeeze(samples['actions'])
	next_obs = np.squeeze(samples['next_observations'])
	rew = np.squeeze(samples['rewards'])
	terms = np.squeeze(samples['terminals'])

	delta_obs = next_obs - obs

	#### ----END preprocess samples for model training in safety gym -----####
	if priors is not None:
		inputs = np.concatenate((obs, act, priors), axis=-1)
	else:
		inputs = np.concatenate((obs, act), axis=-1)

	outputs = np.concatenate((delta_obs, rew[..., None]), axis=-1)

	# add noise
	if noise:
		inputs = _add_noise(inputs, noise)		### noise helps 

	return inputs, outputs

def format_samples_for_cost(samples, oversampling=False, one_hot = True, num_classes=2, priors = None, noise=None):
	"""
	formats samples to fit training for cost, specifically returns: 

	if priors are given, they will be concatenated to the inputs. 
	priors have to be the same length in the first dimension as the samples
	given (that is, the number of samples)

	Currently only uses next_obs for inputs of the cost, implying: C = f(s')

	Args:
		one_hot: determines whether targets are structured as classification or regression
					one_hot: True will output targets with shape [batch_size, num_classes]
					one_hot: False wil output targets with shape [batch_size,] and scalar targets
	"""
	next_obs = np.squeeze(samples['next_observations'])
	obs = np.squeeze(samples['observations'])
	cost = samples['costs']
	act = np.squeeze(samples['actions'])

	if one_hot:
		cost_one_hot = np.zeros(shape=(len(cost), num_classes))
		batch_indcs = np.arange(0, len(cost))
		costs = cost.astype(int)
		cost_one_hot[(batch_indcs, costs)] = 1
		outputs = cost_one_hot
	else:
		outputs = cost[:, None]

	if priors is not None:
		inputs = np.concatenate((obs, act, next_obs, priors), axis=-1)
	else:
		inputs = np.concatenate((obs, act, next_obs), axis=-1)

	
	## ________________________________ ##
	##      oversample cost classes     ##
	## ________________________________ ##
	if oversampling:
		if len(outputs[np.where(costs>0)[0]])>0:
			imbalance_ratio = len(outputs[np.where(costs==0)[0]])//len(outputs[np.where(costs>0)[0]])
			extra_outputs = np.tile(outputs[np.where(costs>0)[0]], (1+imbalance_ratio//3,1))		## don't need to overdo it
			outputs = np.concatenate((outputs, extra_outputs), axis=0)
			extra_inputs = np.tile(inputs[np.where(costs>0)[0]], (1+imbalance_ratio//3,1))
			extra_inputs = _add_noise(extra_inputs, 0.0001)
			inputs = np.concatenate((inputs, extra_inputs), axis=0)
	
	### ______ add noise _____ ###
	if noise:
		inputs = _add_noise(inputs, noise)		### noise helps 
	
	return inputs, outputs

def _add_noise(data_inp, noiseToSignal):
    data= copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis = 0)
    std_of_noise = mean_data*noiseToSignal
    for j in range(mean_data.shape[0]):
        if(std_of_noise[j]>0):
            data[:,j] = np.copy(data[:,j]+np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
    return data


def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))

if __name__ == '__main__':
	model = construct_model()
