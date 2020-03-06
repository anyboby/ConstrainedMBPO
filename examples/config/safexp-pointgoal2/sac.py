params = {
    'type': 'MBPO',
    'universe': "gym",
    'domain': "Safexp-PointGoal2",
    'task': "v0",

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',
    'use_mjc_state_model': False,  
    'preprocessing_type':'Safexp-PointGoal2',
    

    #'max_pool_size':int(5e5),         # haven't seen oom in pointgoal so far

    'kwargs': {
        'epoch_length': 3000, #1000,    # refers to how many samples (one obs per sample usually) are collected in one epoch
        'train_every_n_steps': 50,       # Repeat training of rl_algo n_train_repeat times every _train_every_n_steps 
        'n_train_repeat': 10, #40,      # -> refers to total timesteps
        'eval_render_mode': None,    # 
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_std_inc' : 0.02,         # only relevant if use_mjc_state_model is True

        'hidden_dim':270,               # hidden layer size of model bnn
        'model_train_freq': 250,
        'model_retain_epochs': 5,
        'rollout_batch_size': 100e3,    # how does this relate to real_ratio ?
        'deterministic': False,                     
        'num_networks': 7,              # size of model network ensemble
        'num_elites': 5,                # best networks to select from num_networks
        'real_ratio': 1,#0.05,          # how many rollouts compared to real rollouts
        'target_entropy': -3,
        'max_model_t': None,
        'rollout_schedule': [20, 150, 1, 15],   # min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
                                                # increases rollout length from min_length to max_length over 
                                                # range of (min_epoch, max_epoch)
    }
}

