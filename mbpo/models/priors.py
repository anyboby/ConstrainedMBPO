import numpy as np


# ========= PRIORS for dynamics model =========#
# include prior information that your dynamics model 
# doesn't capture in desirable quality here here.


def prior_safety_gym(obs, acts, infos):

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
    if infos is not None:
        acc_spikes = np.concatenate((acc_spikes, infos), axis=-1)
    return acc_spikes


PRIORS_BY_DOMAIN={
    'Safexp-PointGoal2':prior_safety_gym,
    'Safexp-PointGoal0':prior_safety_gym,
    'Safexp-PointGoal1':prior_safety_gym,
}

PRIOR_DIMS = {
    'Safexp-PointGoal2':46,
    'Safexp-PointGoal0':6, #?
    'Safexp-PointGoal1':24,
}
