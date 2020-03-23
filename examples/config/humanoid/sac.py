params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'Humanoid',
    'task': 'v2',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 3000,
        'train_every_n_steps': 50,
        'n_train_repeat': 10,
        'eval_render_mode': 'human',
        'eval_n_episodes': 5,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 1000,
        'model_retain_epochs': 5,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 1,
        'target_entropy': -3,
        'max_model_t': None,
        'rollout_schedule': [20, 300, 1, 20],
        'hidden_dim': 400,
    }
}
