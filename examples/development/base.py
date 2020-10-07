from ray import tune
import numpy as np
import pdb

from softlearning.misc.utils import get_git_rev, deep_update

M = 256 #256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Pendulum': 200,
}


GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

CPO_POLICY_PARAMS_BASE = {
    'type': 'CPOPolicy',
    'kwargs': {
        'a_hidden_layer_sizes':   (M, M),
        'squash': True,
        'dyn_ensemble_size':    tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['algorithm_params']['kwargs']['num_networks'] 
            )),
        'vf_lr':                3e-4,
        'vf_hidden_layer_sizes':(128,128), #(128, 128, 128, 128),
        'vf_epochs':            10,                 
        'vf_batch_size':        2048,
        'vf_ensemble_size':     7,
        'vf_elites':            5,
        'vf_activation':        'swish',
        'vf_loss':              'MSE',          # choose from #'NLL' (inc. var); 'MSE' ; 'Huber'
        'vf_decay':             1e-6,
        'vf_clipping':          False,           # clip losses for a trust-region like update
        'vf_kl_cliprange':      0.0,
        'vf_var_corr':          False,           # include variance correction terms acc. to paper, only use with NLL
        'v_logit_bias':         1.0,#1,         # logit bias to control initial values
        'vc_logit_bias':        1.0,# 10,
        'ent_reg':              0.0,
        'target_kl':            0.1,
        'cost_lim_end':         25e3,
        'cost_lim':             25e3,
        'cost_lam':             .98,
        'cost_gamma':           0.97,
        'lam':                  .95,
        'gamma':                0.99,
        'epoch_length': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['algorithm_params']['kwargs']['epoch_length'] 
            )),
        'max_path_length': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['sampler_params']['kwargs']['max_path_length']
            )),
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}
CPO_POLICY_PARAMS_FOR_DOMAIN = {}


POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
    'CPOPolicy' :   CPO_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
    'cpopolicy': POLICY_PARAMS_BASE['CPOPolicy']
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
    'CPOPolicy': CPO_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
    'cpopolicy': POLICY_PARAMS_FOR_DOMAIN['CPOPolicy'],
})

ALGORITHM_PARAMS_ADDITIONAL = {
    'MBPO': {
        'type': 'MBPO',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000), #5000
        }
    },
    'CMBPO': {
        'type': 'CMBPO',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(10000), #5000
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    },
    'MVE': {
        'type': 'MVE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    # 'Swimmer': int(3e3),
    # 'Hopper': int(1e3),
    # 'HalfCheetah': int(3e3),
    # 'Walker2d': int(3e3),
    # 'Ant': int(3e3),
    # 'Humanoid': int(1e4),
    'Pusher2d': int(2e3),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(200),
    'Reacher': int(200),
    'Pendulum': 10,
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENVIRONMENT_PARAMS = {
    'Swimmer': {  # 2 DoF
    },
    'Hopper': {  # 3 DoF
    },
    'HalfCheetah': {  # 6 DoF
    },
    'Walker2d': {  # 6 DoF
    },
    'Ant': {  # 8 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Humanoid': {  # 17 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Pusher2d': {  # 3 DoF
        'Default-v3': {
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 1.0,
            'goal': (0, -1),
        },
        'DefaultReach-v0': {
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'ImageDefault-v0': {
            'image_shape': (32, 32, 3),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 3.0,
        },
        'ImageReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'BlindReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        }
    },
    'Point2DEnv': {
        'Default-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
        'Wall-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
    }
}

NUM_CHECKPOINTS = 10

REPLAY_POOL_PARAMS_PER_ALGO = {
    'default': {
        'type': 'SimpleReplayPool',
        'preprocess_type': 'default',
        'kwargs': {
            'max_size': tune.sample_from(lambda spec: (
                {
                    'SimpleReplayPool': int(1e6),
                    'TrajectoryReplayPool': int(1e4),
                    'CPOBuffer':int(6e4),
                }.get(
                    spec.get('config', spec)
                    ['replay_pool_params']['type'],
                    int(1e6))
            )),
        }
    },
    'CMBPO': {
        'type': 'CPOBuffer',
        'preprocess_type': 'default',
        'kwargs': {
            'size': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['algorithm_params']['kwargs']['epoch_length'] 
            )),
            'archive_size': tune.sample_from(lambda spec: (
                {
                    'SimpleReplayPool': int(1e6),
                    'TrajectoryReplayPool': int(1e4),
                    'CPOBuffer':int(10e4),
                }.get(
                    spec.get('config', spec)
                    ['replay_pool_params']['type'],
                    int(1e6))
            )),
            'value_ensemble_size': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['policy_params']['kwargs'].get('vf_ensemble_size',1)
            )),
            'rollout_mode': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['algorithm_params']['kwargs'].get('rollout_mode',False)
            )),


        }
    },
}

SAMPLER_TYPES_PER_ALGO = {
    'default': 'SimpleSampler',
    'CMBPO': 'CPOSampler',
}


def get_variant_spec_base(universe, domain, task, policy, algorithm, env_params):
    algorithm_params = deep_update(
        env_params,
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    )
    algorithm_params = deep_update(
        algorithm_params,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    variant_spec = {
        'git_sha': get_git_rev(),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {}),
            {'log_dir':env_params['log_dir']},
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': REPLAY_POOL_PARAMS_PER_ALGO.get(algorithm, REPLAY_POOL_PARAMS_PER_ALGO['default']),
        'sampler_params': {
            'type': SAMPLER_TYPES_PER_ALGO.get(algorithm, SAMPLER_TYPES_PER_ALGO['default']),
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
                'preprocess_type': 'default'#'default'#'pointgoal0'
            },
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(    #@anyboby uncomment
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            # 'checkpoint_frequency': 1,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec

def get_variant_spec(args, env_params):
    universe, domain, task = env_params.universe, env_params.domain, env_params.task

    variant_spec = get_variant_spec_base(
        universe, domain, task, args.policy, env_params.type, env_params)


    # overwrite some manually inserted params
    if 'max_pool_size' in env_params:
        variant_spec['replay_pool_params']['kwargs']['max_size'] = env_params.max_pool_size

    if 'use_mjc_state_model' in env_params and env_params.use_mjc_state_model:
        variant_spec['replay_pool_params']['type'] = 'MjcStateReplayPool'
        variant_spec['sampler_params']['type'] = 'MjcStateSampler'
        variant_spec['algorithm_params']['kwargs']['use_mjc_state_model'] = True
    else:
        variant_spec['algorithm_params']['kwargs']['use_mjc_state_model'] = False
        

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)


    return variant_spec
