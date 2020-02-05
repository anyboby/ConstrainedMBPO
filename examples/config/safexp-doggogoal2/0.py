params = {
    'type': 'MBPO',
    'universe': "gym",
    'domain': "Safexp-DoggoGoal2",
    'task': "v0",

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 3000, #1000,    # refers to how many samples (one obs per sample usually) are collected in one epoch
        'train_every_n_steps': 100,     # Repeat training n_train_repeat times every _train_every_n_steps
        'n_train_repeat': 10, #40,      # -> trains on current epochs training batch, every_n_steps
        'eval_render_mode': 'human',    # 
        'eval_n_episodes': 5,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,    # how does this relate to real_ratio ?
        'deterministic': False,                     
        'num_networks': 7,              # size of model network ensemble
        'num_elites': 5,                # best networks to select from num_networks
        'real_ratio': 1,#0.05,          # how many rollouts compared to real rollouts
        'target_entropy': -1,
        'max_model_t': None,
        'rollout_schedule': [20, 150, 1, 15],
    }
}

