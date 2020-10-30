#!/usr/bin/env python

import time
import numpy as np
import os
from softlearning.policies.safe_utils.logx import load_policy
from softlearning.policies.safe_utils.logx import EpochLogger
from mbpo.models import fake_env, bnn, constructor
import tensorflow as tf
import matplotlib.pyplot as plt

import time
import random
 
def run_policy(env, get_action, fpath, max_ep_len=None, num_episodes=10, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    params1= {
            'name':'BNN_10262',
            'load_model':True,
            'model_dir':'/home/mo/ray_mbpo/AntSafe/defaults/seed:2608_2020-10-30_02-55-39h74qa9rw/checkpoint_340/',
            'num_networks':7,
            'num_elites':5,
            'loss':'MSE',
            'use_scaler_in':True,
            'use_scaler_out':True,
            }
    model1 = bnn.BNN(params1)

    params2= {
            'name':'BNN_923',
            'load_model':True,
            'model_dir':'/home/mo/ray_mbpo/AntSafe/defaults/seed:2608_2020-10-30_02-55-39h74qa9rw/checkpoint_40/',
            'num_networks':7,
            'num_elites':5,
            'loss':'MSE',
            'use_scaler_in':True,
            'use_scaler_out':True,
            }
    model2 = bnn.BNN(params2)


    model2.finalize(tf.train.AdamOptimizer)
    model1.finalize(tf.train.AdamOptimizer)
                                            
    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    xdata = []
    ydata1 = []
    ydata2 = []
    plt.show()
    
    axes = plt.gca()
    axes.set_xlim(0, 700)
    axes.set_ylim(0, 0.1)
    axes.axvline(x=150, ymin=0, ymax=10, color='black', linewidth=2.5)
    axes.axvline(x=350, ymin=0, ymax=10, color='black', linewidth=2.5)
    axes.set_title('Ensemble Disagreement')
    # axes.set_yscale('log')
    line1, = axes.plot(xdata, ydata1, color='green', label='Model trained under current policy') #'r-')
    line2, = axes.plot(xdata, ydata2, color='red', label='Model trained under older policy')#'r-')
    axes.legend(fontsize='x-large', loc='upper left')

    while n < num_episodes:
        if render:
            env.render(mode='human')
            # time.sleep(1e-3)
        # @TODO anyboby
        ### bad hardcoding
        o_main = o[:29][None]
        
        a_main = get_action(o_main)

        if ep_len==150 or ep_len==350:
            time.sleep(3)
            m_steps = 0
            m_obs1 = o_main
            m_obs2 = o_main
            while True:
                ep_len += 1
                m_steps += 1

                a_m1 = get_action(m_obs1)


                ######### model 1
                inputs1 = np.concatenate([m_obs1,a_m1], axis=-1)
                n_obs1 = model1.predict(inputs1, factored=True, inc_var=False)

                unc1 = np.mean(np.var(n_obs1, axis=0))

                n_obs1 = n_obs1[np.random.randint(7)]
                n_obs1[...,:-1] += m_obs1
                n_obs_qpos1 = n_obs1[...,:13]
                n_obs_qvel1 = n_obs1[...,13:27]
                n_obs_x1 = n_obs1[...,27:28]*5
                n_obs_y_off1 = n_obs1[...,28:29]

                if n_obs_x1<20:
                    n_obs_y1 = n_obs_y_off1 + n_obs_x1*np.tan(30/360*2*np.pi)
                elif n_obs_x1>20 and n_obs_x1<60:
                    n_obs_y1 = n_obs_y_off1 - (n_obs_x1-40)*np.tan(30/360*2*np.pi)
                elif n_obs_x1>60 and n_obs_x1<100:
                    n_obs_y1 = n_obs_y_off1 + (n_obs_x1-80)*np.tan(30/360*2*np.pi)
                else:
                    n_obs_y1 = n_obs_y_off1 + 20*np.tan(30/360*2*np.pi)

                ########## model 2
                a_m2 = get_action(m_obs2)
                inputs2 = np.concatenate([m_obs2,a_m2], axis=-1)

                n_obs2 = model2.predict(inputs2, factored=True, inc_var=False)

                unc2 = np.mean(np.var(n_obs2, axis=0))

                n_obs2 = n_obs2[np.random.randint(7)]
                n_obs2[...,:-1] += m_obs2
                n_obs_qpos2 = n_obs2[...,:13]
                n_obs_qvel2 = n_obs2[...,13:27]
                n_obs_x2 = n_obs2[...,27:28]*5
                n_obs_y_off2 = n_obs2[...,28:29]

                if n_obs_x2<20:
                    n_obs_y2 = n_obs_y_off2 + n_obs_x2*np.tan(30/360*2*np.pi)
                elif n_obs_x2>20 and n_obs_x2<60:
                    n_obs_y2 = n_obs_y_off2 - (n_obs_x2-40)*np.tan(30/360*2*np.pi)
                elif n_obs_x2>60 and n_obs_x2<100:
                    n_obs_y2 = n_obs_y_off2 + (n_obs_x2-80)*np.tan(30/360*2*np.pi)
                else:
                    n_obs_y2 = n_obs_y_off2 + 20*np.tan(30/360*2*np.pi)
                ###########


                xdata.append(ep_len)
                ydata1.append(unc1)
                ydata2.append(unc2)
                line1.set_xdata(xdata)
                line1.set_ydata(ydata1)
                line2.set_xdata(xdata)
                line2.set_ydata(ydata2)

                plt.draw()
                plt.pause(1e-17)

                a = get_action(o_main)
                
                o, _, _, _ = env.step(np.concatenate([a, a_m1, a_m2], axis=-1))
                o_main=o[:29][None]
                real_qpos = o_main[...,:13]
                real_qvel = o_main[...,13:27]
                real_x = o_main[...,27:28]*5
                real_y_off = o_main[...,28:29]
                if real_x<20:
                    real_y = real_y_off + real_x*np.tan(30/360*2*np.pi)
                elif real_x>20 and real_x<60:
                    real_y = real_y_off - (real_x-40)*np.tan(30/360*2*np.pi)
                elif real_x>60 and real_x<100:
                    real_y = real_y_off + (real_x-80)*np.tan(30/360*2*np.pi)
                else:
                    real_y = real_y_off + 20*np.tan(30/360*2*np.pi)


                qpos = np.concatenate([real_x, real_y, real_qpos, n_obs_x2, n_obs_y2, n_obs_qpos2, n_obs_x1, n_obs_y1, n_obs_qpos1], axis=-1)
                qvel = np.concatenate([real_qvel, n_obs_qvel2, n_obs_qvel1], axis=-1)

                init_qpos = env._env.env.init_qpos.copy()
                init_qpos[:-42] = qpos
                init_qvel = env._env.env.init_qvel.copy()
                init_qvel[:-36] = qvel

                env._env.env.set_state(qpos=init_qpos, qvel=init_qvel)
                env.render(mode='human')

                m_obs1 = n_obs1[...,:-1]
                m_obs2 = n_obs2[...,:-1]

                if m_steps > 198:
                    # time.sleep(3)
                    break
                

        pred1 = model1.predict(np.concatenate((o_main,a_main), axis=-1), factored=True, inc_var=False)
        unc1 = np.mean(np.var(pred1, axis=0))

        pred2 = model2.predict(np.concatenate((o_main,a_main), axis=-1), factored=True, inc_var=False)
        unc2 = np.mean(np.var(pred2, axis=0))

        o, r, d, info = env.step(np.concatenate([a_main, a_main, a_main], axis=-1))#a)
        ep_ret += r #r[0]
        ep_cost +=  info.get('cost',0) #info[0].get('cost', 0)
        ep_len += 1


        #### restore positions
        o_main=o[:29][None]
        real_qpos = o_main[...,:13]
        real_qvel = o_main[...,13:27]
        real_x = o_main[...,27:28]*5
        real_y_off = o_main[...,28:29]
        if real_x<20:
            real_y = real_y_off + real_x*np.tan(30/360*2*np.pi)
        elif real_x>20 and real_x<60:
            real_y = real_y_off - (real_x-40)*np.tan(30/360*2*np.pi)
        elif real_x>60 and real_x<100:
            real_y = real_y_off + (real_x-80)*np.tan(30/360*2*np.pi)
        else:
            real_y = real_y_off + 20*np.tan(30/360*2*np.pi)


        qpos = np.concatenate([real_x, real_y, real_qpos, real_x, real_y, real_qpos, real_x, real_y, real_qpos], axis=-1)
        qvel = np.concatenate([real_qvel, real_qvel, real_qvel], axis=-1)

        init_qpos = env._env.env.init_qpos.copy()
        init_qpos[:-42] = qpos
        init_qvel = env._env.env.init_qvel.copy()
        init_qvel[:-36] = qvel

        env._env.env.set_state(qpos=init_qpos, qvel=init_qvel)

        #### plot
        xdata.append(ep_len)
        ydata1.append(unc1)
        ydata2.append(unc2)
        line1.set_xdata(xdata)
        line1.set_ydata(ydata1)
        line2.set_xdata(xdata)
        line2.set_ydata(ydata2)
        plt.draw()
        plt.pause(1e-17)
        # time.sleep(0.1)


        if d or (ep_len >= 500): #(ep_len == max_ep_len):
            #logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1
            xdata = []
            ydata1 = []
            ydata2 = []



    # logger.log_tabular('EpRet', with_min_and_max=True)
    # logger.log_tabular('EpCost', with_min_and_max=True)
    # logger.log_tabular('EpLen', average_only=True)
    # logger.dump_tabular(output_dir = os.path.join(fpath, '/Inference'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=1000)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.fpath, args.len, args.episodes, not(args.norender))
