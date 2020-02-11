params = {
    'type': 'MBPO',
    'universe': "gym",
    'domain': "Safexp-DoggoGoal2",
    'task': "v0",

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'max_pool_size':int(6e5),           # doggo might go oom at 7.5e5

    'kwargs': {
        'epoch_length': 1000, #1000,    # refers to how many samples (one obs per sample usually) are collected in one epoch
        'train_every_n_steps': 1,      # Repeat training n_train_repeat times every _train_every_n_steps
        'n_train_repeat': 40, #40,       # -> trains on current epochs training batch, every_n_steps
        'eval_render_mode': 'human',   # 
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'hidden_dim':420,               # hidden layer size of model bnn
        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,    # how does this relate to real_ratio ?
        'deterministic': False,                     
        'num_networks': 7,              # size of model network ensemble
        'num_elites': 5,                # best networks to select from num_networks
        'real_ratio': 0.05,          # how many rollouts compared to real rollouts
        'target_entropy': -3,
        'max_model_t': None,
        'rollout_schedule': [20, 150, 1, 15],
    }
}

