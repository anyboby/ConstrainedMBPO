params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'AntSafe',
    'task': 'v2',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 10000,
        'train_every_n_steps': 3,
        'n_train_repeat': 20,
        'eval_render_mode': None,
        'eval_n_episodes': 3,
        'eval_deterministic': False,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 1000,
        'model_retain_epochs': 1,
        'rollout_batch_size': 20e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -4,
        'max_model_t': 50,
        'rollout_schedule': [20, 150, 1, 1],
    }
}
