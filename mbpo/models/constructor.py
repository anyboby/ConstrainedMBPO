import numpy as np
import tensorflow as tf

from mbpo.models.fc import FC
from mbpo.models.bnn import BNN

def construct_model(obs_dim_in=11, obs_dim_out=None, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):
	if not obs_dim_out:
		obs_dim_out = obs_dim_in

	print('[ BNN ] Observation dim in / out: {} / {} | Action dim: {} | Hidden dim: {}'.format(obs_dim_in, obs_dim_out, act_dim, hidden_dim))
	params = {'name': 'BNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
	model = BNN(params)

	model.add(FC(hidden_dim, input_dim=obs_dim_in+act_dim, activation="swish", weight_decay=0.000015))	#0.000025))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.00002))			#0.00005))
	#model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))		#@anyboby optional
	#model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))		#@anyboby optional
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))		#0.000075))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))		#0.000075))
	model.add(FC(obs_dim_out+rew_dim, weight_decay=0.00005))							#0.0001
	model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
	return model

def format_samples_for_training(samples, stacks=1, stacking_axis=0):
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
	if stacks>1:
		unstacked_obs_size = int(obs.shape[1+stacking_axis]/stacks)
		delta_obs = next_obs[:, -unstacked_obs_size:] - obs[:, -unstacked_obs_size:]
	else: 
		delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((delta_obs, rew), axis=-1)
	return inputs, outputs


def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))

if __name__ == '__main__':
	model = construct_model()
