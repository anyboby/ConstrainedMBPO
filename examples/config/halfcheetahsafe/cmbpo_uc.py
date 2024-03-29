params = {
    'type': 'CMBPO',
    'universe': 'gym',
    'domain': 'HalfCheetahSafe',
    'task': 'v2',

    'policy':'CPOPolicy',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',
    'use_mjc_state_model': False,      

    'kwargs': {
        'n_epochs': 10000000,        ### this is only a placeholder for now, n_env_interacts terminates properly
        'n_env_interacts': 5000000,    
        'epoch_length': 50000, #1000,    # samples per epoch, also determines train frequency 
        'train_every_n_steps': 1,       # Repeat training of rl_algo n_train_repeat times every _train_every_n_steps 
        'n_train_repeat': 1, #20 #40,      # -> refers to total timesteps
        'eval_render_mode': None,    # 
        'eval_n_episodes': 0,
        'eval_every_n_steps': 50e3,
        'eval_deterministic': False,    # not implemented in cmbpo

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        #'model_std_inc' : 0.02,         # only relevant if use_mjc_state_model is True

        #### it is crucial to choose a model that doesn't overfit when trained too often on seen data
        ## for model architecture finding:  1. play around with the start samples to find an architecture, that doesn't really overfit
                                          # 2. _epochs_since_update in the bnn can somewhat limit overfitting, but is only treating the symptom
                                          # 3. try finding a balance between the size of new samples per number of
                                          #  updates of the model network (with model_train_freq)

        'hidden_dims':(512,512), #(512, 512, 512, 512),               # hidden layer size of model bnn
        'model_train_freq': 4000,        # model is only trained every (self._timestep % self._model_train_freq==0) steps (terminates when stops improving)
        'model_retain_epochs': 1,       # how many rollouts over the last epochs should be retained in the model_pool (therefore directly affects model_pool size)
        'rollout_batch_size': 1.0e3,    # rollout_batch_size is the size of randomly chosen states to start from when rolling out model
        'deterministic': False,          
        'num_networks': 7,              # size of model network ensemble
        'num_elites': 5,                # best networks to select from num_networks
        'real_ratio': 0.05, #0.05,      # ratio to which the training batch for the rl_algo is composed
        'target_entropy': -3, 
        'max_model_t': None,            # a timeout for model training (e.g. for speeding up wallclock time)
        'dyn_model_train_schedule': [50, 100, 1, 1],
        'cost_model_train_schedule': [25, 80, 1, 1],
        'cares_about_cost': True,
        'policy_alpha': 2,           
        'max_uncertainty_c' :4.0,              ### only applies if rollout_mode=='iv_gae' or rollout_mode=='uncertainty'
        'max_uncertainty_rew' : 3.5,
        'rollout_mode' : 'uncertainty',           #### choose from 'iv_gae', 'schedule', or 'uncertainty'
        'rollout_schedule': [100, 2500, 5, 5], #[15, 100, 1, 15],    # min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
                                                    # increases rollout length from min_length to max_length over 
                                                    # range of (min_epoch, max_epoch)
                                                    ### Only applies if rollout_mode=='schedule'
        'maxroll': 25,      ### only really relevant for iv gae
        'max_tddyn_err' : 0.6,
        'max_tddyn_err_decay' : 1,
        'batch_size_policy': 50000,              ### how many samples 
        'min_real_samples_per_epoch': 15000,
    }
}