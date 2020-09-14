import numpy as np


# ========= PRIORS for dynamics model =========#
# include prior information that your dynamics model 
# doesn't capture in desirable quality here here.


def prior_safety_gym(obs, acts):

    ## accleration spikes
    def process_act(act, last_act):
        '''
        Predicts a spike based on 0-transition between actions
        !! very specifically designed for x-acceleration spike detection
        returns a normalized prediction signal for y-acceleration in mujoco envs
        a shape (1,) np array
        
        '''
        act_x = act
        last_act_x = last_act
        acc_spike = 0
        ### acc
        if last_act_x==act_x:
            acc_spike=0
        else:
            if last_act_x<=0<=act_x or act_x<=0<=last_act_x:
                #pass
                acc_spike = act_x-last_act_x
                acc_spike = acc_spike/abs(acc_spike) #normalize
        return acc_spike

    last_acts = obs[...,:2]

    process_act_vec = np.vectorize(process_act)
    #last_act = np.concatenate((np.zeros_like(acts[0])[np.newaxis], acts[:-1,:]), axis=0)

    acc_spikes = process_act_vec(acts, last_acts)
    return acc_spikes

def post_safety_gym(obs, acts):
    assert len(obs.shape) == len(acts.shape)
    if len(obs.shape)==1:
        obs = obs[None]
        acts = acts[None]
        return_single = True
    else: return_single = False

    obs[...,:2] = acts[...,:2]
    if return_single:
        return obs[0] 
    else: 
        return obs

def safety_gym_weights(obs_dim):
    # loss weights for dyn model. Set manually
    mse_weights = np.ones(obs_dim+1, dtype='float32') # +2 for rew and costs
    mse_weights[0:2] = 1    # predicting prev action given the action should be easy, but has very high unnormalized target values. 
    mse_weights[3:19]=1 # goal lidar
    mse_weights[2]=1   # goal dist.
    mse_weights[-1]= 1   # rew function

    return mse_weights


PRIORS_BY_DOMAIN={
    'Safexp-PointGoal2':prior_safety_gym,
    'Safexp-PointGoal0':prior_safety_gym,
    'Safexp-PointGoal1':prior_safety_gym,
}

PRIOR_DIMS = {
    'Safexp-PointGoal2':2,
    'Safexp-PointGoal0':2,
    'Safexp-PointGoal1':2,
}

POSTS_BY_DOMAIN={
    # 'Safexp-PointGoal2':post_safety_gym,
    'Safexp-PointGoal0':post_safety_gym,
    'Safexp-PointGoal1':post_safety_gym,
}

WEIGHTS_PER_DOMAIN = {
    # 'Safexp-PointGoal2':safety_gym_weights,
    'Safexp-PointGoal1':safety_gym_weights,
    'Safexp-PointGoal0':safety_gym_weights,
}
