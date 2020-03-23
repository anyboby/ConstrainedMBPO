import numpy as np
import numpy.ma as ma
import tensorflow as tf

import copy
from mbpo.models.fc import FC
from mbpo.models.bnn import BNN

def construct_model(obs_dim_in=11, 
					obs_dim_out=None, 
					act_dim=3, 
					rew_dim=1, 
					hidden_dim=200, 
					num_networks=7, 
					num_elites=5, 
					weighted=False, 
					session=None):
	if not obs_dim_out:
		obs_dim_out = obs_dim_in

	print('[ BNN ] Observation dim in / out: {} / {} | Action dim: {} | Hidden dim: {}'.format(obs_dim_in, obs_dim_out, act_dim, hidden_dim))
	params = {'name': 'BNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
	model = BNN(params)

	model.add(FC(hidden_dim, input_dim=obs_dim_in+act_dim, activation="swish", weight_decay=0.000025))	#0.000025))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))			#0.00005))
	#model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))		#@anyboby optional
	#model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))		#@anyboby optional
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))		#0.000075))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))		#0.000075))
	model.add(FC(obs_dim_out+rew_dim, weight_decay=0.0001))							#0.0001
	model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001}, weighted=weighted, )
	return model

def format_samples_for_training(samples, safe_config=None, add_noise=False):
	"""
	formats samples to fit training, specifically returns: 

	inputs, outputs:

	inputs = np.concatenate((observations, act), axis=-1)
	outputs = np.concatenate((delta_observations, rewards), axis=-1)

	"""
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']

	#### ---- preprocess samples for model training in safety gym -----####
	if safe_config:
		stacks = safe_config['stacks']
		stacking_axis = safe_config['stacking_axis']
		unstacked_obs_size = int(obs.shape[1+stacking_axis]/stacks)	
		delta_obs = next_obs[:, -unstacked_obs_size:] - obs[:, -unstacked_obs_size:]
		
		## remove terminals and outliers, otherwise they will confuse the model when close to a goal:
		outlier_threshold = 0.2
		mask = np.invert(samples['terminals'][:,0])
		mask_outlier = np.invert(np.max(ma.masked_greater(abs(delta_obs[:,:16]), outlier_threshold).mask, axis=-1))  ###@anyboby for testing, code this better tomorrow !
		mask = mask*mask_outlier

		obs=obs[mask]
		act=act[mask]
		next_obs=next_obs[mask]
		rew = rew[mask]
		delta_obs = delta_obs[mask]			## testing, for similar gradient magnitudes
		# delta_obs = np.clip(delta_obs,-outlier_threshold, outlier_threshold)		## clip delta_obs

	else: 
		delta_obs = next_obs - obs
	#### ----END preprocess samples for model training in safety gym -----####
	
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((delta_obs, rew), axis=-1)		###@anyboby testing

	# add noise
	if add_noise:
		inputs = _add_noise(inputs, 0.0001)		### noise helps 


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
