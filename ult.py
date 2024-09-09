from scipy.stats import norm
import numpy as np
import multiprocess
import torch
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from numpy import pi

def calculate_mu_sigma(data):
    mu = np.mean(data)
    sigma = np.std(data, ddof=0)
    return mu, sigma


def probability_of_zero(mu, sigma):
    '''    Probability density at x = 0'''
    prob_density_0 = norm.pdf(0, loc=mu, scale=sigma)
    return prob_density_0


def run_one_episode(env, model, deterministic=True):
    '''run one ep.'''

    actions, rewards, states, observations, beliefs,uncertainty = [], [], [], [], [],[]

    belief=env.reset()
    states.append([x.item() for x in env.x])
    beliefs.append([x.item() for x in env.z[0]]) # env.z[0] is belief mu.
    uncertainty.append(env.z[1]) # env.z[1] is belief cov.

    t=0
    while True:
        action, _ = model.predict(belief, deterministic=deterministic)
        actions.append(action.item())
        belief, reward, done, info = env.step(action)
        rewards.append(reward.item())
        states.append([x.item() for x in env.x])
        observations.append([x.item() for x in info['obs']])
        beliefs.append([x.item() for x in env.z[0]]) # env.z[0] is belief mu.
        uncertainty.append(env.z[1]) # env.z[1] is belief cov.

        t += 1
        if done:
            break

    obs_phase_f,obs_phase_w = env.obs_phase_save
    episode = {
        'num_steps': t, # int
        'actions': np.array(actions),  # [0, t)
        'rewards': np.array(rewards),  # [0, t)
        'states': np.array(states),  # [0, t]
        'observations': np.array(observations),  # [0, t]
        'beliefs': np.array(beliefs),  # [0, t]
        'obs_phase_f':obs_phase_f,
        'obs_phase_w':obs_phase_w,    
        }

    return episode


def logll(true=None, estimate=None, std=0.3, error=None, prob=False):
    '''todo. the likelihood calculation is different? inital belief matters?'''
    # print(error)
    var=std**2
    if error is not None: # use for point eval, obs
        g=lambda x: 1/torch.sqrt(2*pi*torch.ones(1))*torch.exp(-0.5*x**2/var)
        z=1/g(torch.zeros(1)+1e-8)
        loss=torch.log(g(error)*z+1e-8)
    else: # use for distribution eval, aciton
        c=torch.abs(true-estimate)
        gi=lambda x: -(torch.erf(x/torch.sqrt(torch.tensor([2]))/std)-1)/2
        loss=torch.log(gi(c)*2+1e-16)
    if prob:
        return torch.exp(loss)
    return loss


def loglikelihood(agent=None, 
            actions=None, 
            tasks=None, 
            phi=None, 
            theta=None, 
            env=None,
            num_iteration=1, 
            states=None, 
            samples=1, 
            gpu=False,
            action_var=0.1,
            debug=False):
    if gpu:
        logPr = torch.zeros(1).cuda()[0] #torch.FloatTensor([])
    else:
        logPr = torch.zeros(1)[0] #torch.FloatTensor([])
    
    def _wrapped_call(ep, task):     
        logPr_ep = torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]   
        for sample_index in range(samples): 
            mkactionep = actions[ep]
            if mkactionep==[] or mkactionep.shape[0]==0:
                continue
            env.reset(theta=theta, phi=phi, goal_position=task, vctrl=mkactionep[0][0],wctrl=mkactionep[0][1])
            numtime=len(mkactionep[1:])

            # compare mk data and agent actions
            for t,mk_action in enumerate(mkactionep[1:]): # use a t and s t (treat st as st+1)
                # agent's action
                action = agent(env.decision_info)
                # agent's obs, last step obs doesnt matter.
                if t<len(states[ep])-1:
                    if type(states[ep])==list:
                        nextstate=states[ep][1:][t]
                    elif type(states[ep])==torch.Tensor:
                        nextstate=states[ep][1:][t].view(-1,1)
                    else: # np array
                        nextstate=torch.tensor(states[ep])[1:][t].view(-1,1)
                    obs=env.observations(nextstate)
                    # agent's belief
                    env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(mk_action).view(1,-1))
                    previous_action=mk_action # current action is prev action for next time
                    env.trial_timer+=1
                    env.decision_info=env.wrap_decision_info(
                                                previous_action=torch.tensor(previous_action), 
                                                time=env.trial_timer)
                # loss
                action_loss = -1*logll(torch.tensor(mk_action),action,std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss
            # if agent has not stop, compare agent action vs 0,0
            agentstop=torch.norm(action)<env.terminal_vel
            while not agentstop and env.trial_timer<40:
                action = agent(env.decision_info)
                agentstop=torch.norm(action)<env.terminal_vel
                obs=(torch.tensor([0.5,pi/2])*action+env.obs_err()).t()
                env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(action).view(1,-1))
                # previous_action=torch.tensor([0.,0.]) # current action is prev action for next time
                previous_action=action
                env.trial_timer+=1
                env.decision_info=env.wrap_decision_info(
                previous_action=torch.tensor(previous_action), 
                                            time=env.trial_timer)
                # loss
                action_loss = -1*logll(torch.tensor(torch.zeros(2)),action,std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss

        return logPr_ep/samples/env.trial_timer.item()
    
    tik=time.time()
    loglls=[]
    for ep, task in enumerate(tasks):
        logPr_ep=_wrapped_call(ep, task)
        logPr += logPr_ep
        loglls.append(logPr_ep)
        del logPr_ep
    regularization=torch.sum(1/(theta+1e-4))
    print('calculate loss time {:.0f}'.format(time.time()-tik))
    if debug:
        return loglls
    return logPr/len(tasks)+0.01*regularization
