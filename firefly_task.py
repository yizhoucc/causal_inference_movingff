import gym
from gym import spaces
from ult import *
import torch
import torch.nn as nn
from gym.spaces import Discrete, MultiDiscrete, Box
'''
notes about meeting with john 0913
overall flow:
    observation phase:
        we will decide self motion first, then decide obj motion, then calculate the distribution of beliefs.
        1 self motion is very easy to tell. no need to causual inference with math. just assume subject get it 100%
        2 once decide the self motion, we compute the obj motion causual inference, using math equations. note the q threshold should be computed differently given the selfmotion decision.
        note we should compute the esimated v by cov(ft, t)/cov(t,t,) instead of doing avg-avg.
        3 now we have the dicision. there are 4 cases. selfmotion or not, cross objmotion or not. we compute the belief mu and covariance for the case we are in.
    steering phase.
        we can only obs selfmotion
        the control has noise. no system noise.
        obj has acceration in this phase, but we just assume subject know it perfectly (tho learning).
        the optimial strategy is to fully use observation. the w we see, is the noisy action (action+action noise) that act on the system with a deterministic funciton (fixed transition, no system noise.). altho the belief uncertainty about locations will grow due to intergration.


todo:
    add acc, but assume agent know,
    assume agent know self motion.
    fix hg, that is control noise
    fix q given sm or nosm.
    4 cases, compute muand cov differently.


need from john:
script for generating simulation data and optimizing the model.
data and script for loading real data. no need to have all preprocessing, since my goal is to do some visualizations of subject behavior and have better understanding of the task.
updated notes (you mentioned some time ago you are writing one?)

'''



class Model(gym.Env):
    def __init__(self, dt=1, debug=1):
        super().__init__()
        self.dt = dt
        self.t_obs = int(10/dt)
        self.debug = debug
        self.noise_scalar = 0.5
        self.reward_dist = 0.2
        self.timeout = 40  # action phase timeout
        self.observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space=gym.spaces.Box(low=-1,high=1, shape=(1,), dtype=np.float32)

    def _print(self, *message):
        if self.debug:
            print(message)

    def get_random_theta(self,):
        '''randomize a theta for training.'''
        # obs uncertainty
        self.hsig_f = torch.rand(1)*self.noise_scalar  # dist obs unceratinty
        # self motion obs uncertainty
        self.hsig_w = torch.rand(1)*self.noise_scalar
        # prior
        self.hxi_f = torch.rand(1)  # inital distance
        self.hxi_w = torch.rand(1)  # self motion
        self.hxi_v = torch.rand(1)  # obj motion
        self.delta0 = torch.rand(1)  # move or not, decision prior
        self.halpha = torch.rand(1)  # acc?
        self.hg = torch.rand(1)  # control gain
        self.hW_aa = torch.rand(1)
        self.hsig_a = torch.rand(1)

    def apply_theta(self, theta):
        pass

    def get_random_task(self,):
        self.f0 = torch.rand(1)  # inital dist
        if torch.rand(1) > 0.5:
            self.v = torch.rand(1)  # obj motion
        else:
            self.v = torch.zeros(1)
        if torch.rand(1) > 0.5:
            self.w = torch.rand(1)  # self motion
        else:
            self.w = torch.zeros(1)

    def reset(self, theta=None, task=None):
        # task param
        if theta is None:
            self.get_random_theta()
        else:
            self.apply_theta()
        # task condition
        if task is None:
            self.get_random_task()
        else:
            self.apply_task()

        self.t = 0

        # end of obs phase, is start for action phase. f, v, w
        z_0, Sigma_0 = self.obs_phase()  # f, v
        f_0 = self.f_t
        v_0 = self.v
        w_0 = torch.zeros(1)  # start to self control, no self motion.

        self.x = [f_0, v_0, w_0]
        self.z = [z_0, Sigma_0]
        return self.policy_preprocess()

    def obs_dynamic(self,):
        '''return obs sequance of f and w
        f+v-w'''
        # f, w, v=[torch.normal(self.f0,self.hxi_f, size=(1,))],[torch.normal(0,self.hxi_w, size=(1,))],[torch.normal(0,self.hxi_v, size=(1,))]
        # f, w=[torch.normal(self.f0,self.hsig_f, size=(1,))],[torch.normal(self.w,self.hsig_w, size=(1,))]
        f = [self.f0]
        for t in range(self.t_obs):
            new = f[-1]+(self.v-self.w)*self.dt
            f.append(new)
        f = torch.tensor(f[1:])
        self.f_t = f[-1]  # this is the true state.
        for i in range(len(f)):
            f[i] = torch.normal(f[i], self.hsig_f.item())  # add obs noise.
        w = torch.normal(self.w.item(), self.hsig_w.item(), size=(self.t_obs,))

        return f, w

    def decide_selfmotion(self, w):
        '''we can just use the avg w across time, to decide selfmotion.'''
        mu_w = self.t_obs/self.hsig_w**2 / \
            (1/self.hxi_w**2 + self.t_obs/self.hsig_w**2)*torch.mean(w)
        compare = torch.sqrt(nn.functional.relu(torch.log(1+self.hxi_w**2*self.t_obs /
                             self.hsig_w**2) - 2*self.delta0)/(1/self.hxi_w**2 + self.t_obs/self.hsig_w**2))
        self._print(f'decide selfmotion:{abs(mu_w) >= compare}, mu_w {mu_w.item():.2f} > {compare.item():.2f}')
        return abs(mu_w) >= compare

    def decide_objmotion(self, mu_v):
        compare = self.get_threshold_noSM()
        self._print(f'decide object motion:{abs(mu_v) >= compare}, mu_v {mu_v.item():.2f} > {compare.item():.2f}')
        return abs(mu_v) >= compare

    def obs_phase(self,):
        '''obs phase, return unimodel belief.
        also record the current state in self.f_t (distance)'''
        # casaul inference abot self motion and obj motion
        f, w = self.obs_dynamic()
        self.obs_phase_save=[f,w]

        is_selfmotion = self.decide_selfmotion(w)

        time = torch.arange(1, len(f) + 1, dtype=torch.float32)
        time_avg = torch.mean(time)

        f_mean = torch.mean(f)
        numerator = torch.sum((time - time_avg) * (f - f_mean))
        denominator = torch.sum((time - time_avg) ** 2)
        v = numerator / denominator
        mu_w = self.t_obs/self.hsig_w**2 / \
            (1/self.hxi_w**2 + self.t_obs/self.hsig_w**2)*torch.mean(w)
        if is_selfmotion:  # subtract selfmotion
            v = v + mu_w
        self._print(mu_w, 'w estimated')
        self._print(v, 'v estimated')
        is_objmotion = self.decide_objmotion(v)

        # belief cov (2x2), (f, v)
        if is_selfmotion and is_objmotion:
            Sigma = self.get_Sigma_move_SM()
            Sigma = Sigma[[1, 3], :][:, [1, 3]]
        elif is_selfmotion and not is_objmotion:
            Sigma = self.get_Sigma_stat_SM()
            Sigma = Sigma[[1, 3], :][:, [1, 3]]
        elif not is_selfmotion and is_objmotion:
            Sigma = self.get_Sigma_move_noSM()
            Sigma = Sigma[1:, 1:]
        elif not is_selfmotion and not is_objmotion:
            Sigma = self.get_Sigma_stat_noSM()

        covariance=torch.zeros(3,3)
        covariance[:2,:2]=Sigma
        covariance[2,2]=1e-6

        # belief mu
        mu_f = f_mean+self.t_obs/2*(v-mu_w)*self.dt
        mu_z = torch.tensor([mu_f, v, 0.]) # w=0 because obs phase ends.
        return mu_z, covariance

    def policy_preprocess(self,):
        '''process the belief mu and covariance into one vector'''
        zt, St = self.z
        return torch.concat([zt.reshape(-1), St[0, 0].reshape(1), St[1, 1].reshape(1), St[0, 1].reshape(1)])

    def step(self, u):
        '''
        state x: [distance f, invis firefly v, user w]
        belief z: [distance f, invis firefly v, user w] estimation of state
        '''
        

        ft, vt, wt = self.x
        zt, St = self.z
        obs = torch.normal(wt, self.hsig_w)

        ft, vt, wt = self.state_dynamic(ft, vt, wt, u)
        self.x = [ft, vt, wt]
        z_update, S_update = self.belief_dynamic(zt, St, u, obs)
        self.z = [z_update, S_update]

        reward = self.reward()
        # done=False
        # if reward>0 or 
        done = reward > 0. or self.t > self.timeout

        self.t += 1

        agent_observation=self.policy_preprocess()
        info={'obs':obs}
        return agent_observation, reward, done, info

    def reward(self,):
        '''eval reward'''
        ft, vt, wt = self.x
        reach_reward = (abs(ft) <= self.reward_dist)*1
        return reach_reward

    def state_dynamic(self, f, v, w, u):
        '''action phase state dynamic
        u: noisy control'''
        ft = f+(v-w)*self.dt
        wt = u
        return ft, v, wt

    def belief_dynamic(self, z, S, u, obs):
        '''action phase belief dynamic
        z: belief mu. [f,v,w]
        S: belief covariance sigma.
        u: control input.
        x_next = Ax + Bu

        question, why update? no system noise and obs is noisy.
        conlustion: fully use obs. no prediction is neede.
        '''
    
        # prepare
        R = torch.zeros(1, 1)
        R[0, 0] = self.hg**2  # ww todo: a different parameter we need infer
        H = torch.tensor([[0., 0., 1.]])  # f,v,w. can only see w.

        # prediction
        A = self.get_transition_matrix()
        z_next = A@z
        z_next[2] = self.hg*u  # Bu
        S_next = A@S@A.T  # no system noise Q
        # update
        kalman_gain = S_next @ H.T @ np.linalg.inv(H @ S_next @ H.T + R)
        z_update = z_next + kalman_gain @ (obs - H @ z_next)
        S_update = (torch.eye(3) - kalman_gain @ H) @ S_next
        return z_update, S_update

    def get_transition_matrix(self,):
        '''get transition matrix A for action phase
        ordering: f, v, w
        w is directly controlable.'''
        A = torch.zeros(3, 3)
        A[0, 0] = 1  # f integrates on previous
        A[0, 1] = self.dt  # +v
        A[0, 2] = -self.dt  # -w
        A[1, 1] = 1  # v is fixed
        return A

    def getM(self,):
        '''get obs matrix M
        2x3: position, self motion. no way to measure the obj motion'''
        M = torch.zeros(2, 3)
        M[0, 0] = 1
        M[1, 1] = 1
        return M

    def getSigma_obs(self,):
        '''get observation covariance matrix'''
        Sigma = torch.zeros(2, 2)
        Sigma[0, 0] = self.sigma_f**2
        Sigma[1, 1] = self.sigma_w**2
        return Sigma

    def get_threshold_noSM(self):
        '''
        return decision threshold (probability?)
        '''
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        delta0 = self.delta0

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        log_arg = 1 + (xi_v_**2)*(C_t + t2_avg/(1 + xi_f_**2))
        det_ratio = (t_obs/(sig_f**2)) * \
            (C_t + 1/xi_v_**2 + t2_avg/(1 + xi_f_**2))

        thresh_sq = (torch.log(log_arg) - 2*delta0)/det_ratio
        q = torch.sqrt(nn.functional.relu(thresh_sq))
        return q

    # Average posterior covariance, here includes f0 and v but not a

    # checked
    def get_Sigma_move_noSM(self):
        # shape = 3 x 3, a, f, v
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        sig_a = torch.exp(self.hsig_a)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (time**2)/(xi_f_**2)
        prefactor = ((sig_f**2)/t_obs)*(1/(det_inv**2))

        Sig_ff = (C_t + (1/xi_v_**2))**2 + C_t*(time**2)
        Sig_fv = time*(1/((xi_f_*xi_v_)**2) - C_t)
        Sig_vv = (time/(xi_f_**2))**2 + C_t*((1 + 1/xi_f_**2)**2)

        Sig = prefactor * \
            torch.tensor([[0, 0, 0], [0, Sig_ff, Sig_fv], [0, Sig_fv, Sig_vv]])
        Sig[0, 0] = sig_a**2
        return Sig
    # checked

    def get_Sigma_inv_move_noSM(self):
        '''
        sig_f: sigma f, the noise std on observation of firefly position
        sig_a:  ???
        xi_f: prior of firelfy position
        xi_v: prior of firelfy motion
        '''
        # shape = 3 x 3, a, f, v

        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        sig_a = torch.exp(self.hsig_a)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        prefactor = (t_obs/(sig_f**2))*(1/C_t)
        Sig_ff = ((time)/(xi_f_**2))**2 + C_t*((1 + (1/xi_f_)**2)**2)
        Sig_fv = - time*(1/((xi_f_*xi_v_)**2) - C_t)
        Sig_vv = (C_t + (1/xi_v_**2))**2 + C_t*(time**2)

        # Sig = prefactor*torch.tensor([[Sig_ff, Sig_fv, 0],[Sig_fv, Sig_vv, 0],[0,0,0]])
        # Sig[-1,-1] = 1/sig_a**2

        Sig = prefactor*torch.tensor([[0, 0, 0],
                                      [0, Sig_ff, Sig_fv],
                                      [0, Sig_fv, Sig_vv]])
        Sig[0, 0] = 1/sig_a**2
        # result = prefactor*Sig
        return Sig

    # checked; shape = 2x2
    def get_Sigma_stat_noSM(self):
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (time**2)/(xi_f_**2)

        prefactor = (sig_f**2)/t_obs
        Sig_ff = 1/((1 + (1/xi_f_**2))**2)
        Sig_fv = (1/det_inv)*(time/(xi_f_**2 + 1))
        Sig_vv = (1/det_inv**2)*(((time)/(xi_f_**2))
                                 ** 2 + C_t*((1 + (1/xi_f_)**2)**2))

        Sig = torch.tensor([[Sig_ff, Sig_fv], [Sig_fv, Sig_vv]])

        result = prefactor*Sig
        return result

    # checked; shape = 2x2
    def get_Sigma_inv_stat_noSM(self):
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (time**2)/(xi_f_**2)

        prefactor = (t_obs/(sig_f**2))*(1/C_t)
        Sig_ff = ((time)/(xi_f_**2))**2 + C_t*((1 + (1/xi_f_)**2)**2)
        Sig_fv = - det_inv*(time/(xi_f_**2 + 1))
        Sig_vv = (det_inv**2)*(1/((1 + (1/xi_f_**2))**2))

        Sig = torch.tensor([[Sig_ff, Sig_fv], [Sig_fv, Sig_vv]])

        result = prefactor*Sig
        return result

    # checked
    def get_det_Sigma_stat_noSM(self):
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (time**2)/(xi_f_**2)

        prefactor = ((sig_f**2)/t_obs)**2
        result = prefactor*C_t/(det_inv**2)
        return result

    # checked
    def get_det_Sigma_move_noSM(self):
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (time**2)/(xi_f_**2)

        prefactor = ((sig_f**2)/t_obs)**2
        result = prefactor*C_t/(det_inv**2)
        return result

    # Average posterior mean
    # checked
    def get_W_move_noSM(self):
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        W_aa = torch.exp(self.hW_aa)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (time**2)/(xi_f_**2)
        prefactor = 1/det_inv

        W_ff = C_t + 1/xi_v_**2
        W_fv = time/(xi_v_**2)
        W_vf = time/(xi_f_**2)
        W_vv = C_t + t2_avg/(xi_f_**2)

        W = prefactor * \
            torch.tensor([[0, 0, 0], [0, W_ff, W_fv], [0, W_vf, W_vv]])
        W[0, 0] = W_aa
        return W

    # checked
    def get_W_stat_noSM(self):
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - time**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (time**2)/(xi_f_**2)

        W_ff = (1/(1 + 1/xi_f_**2))
        W_fv = (time/(1 + 1/xi_f_**2))
        W_vf = (1/det_inv)*(time/(xi_f_**2))
        W_vv = (1/det_inv)*(C_t + t2_avg/(xi_f_**2))

        W = torch.tensor([[W_ff, W_fv], [W_vf, W_vv]])
        return W

    # checked
    # shapes = (num_t_pts)
    def get_w_xv_A_noSM(self, t, t_obs, t_wait, T):
        alpha = torch.exp(self.halpha)

        denom = 1/(alpha + T - t)
        w_tx = -1/denom
        w_tv = (t_obs + t_wait + T)/denom
        return w_tx, w_tv

    # shape = (2, num_t_pts)
    def get_w_o_move_A_noSM(self, t, t_obs, t_wait, T):
        # get the entire policy
        alpha = torch.exp(self.halpha)

        denom = 1/(alpha + T - t)
        w_f = 1/denom
        w_a = 0.5*(t_wait**2 + T ^ 2)/denom

        w_o = torch.stack((w_a, w_f))
        return w_o

    # shape = (num_t_pts)
    def get_w_o_stat_A_noSM(self, t, t_obs, t_wait, T):
        alpha = torch.exp(self.halpha)

        denom = 1/(alpha + T - t)
        w_f = 1/denom
        return w_f

    # SELF-MOTION STUFF --------------------------------------------
    def get_Sigma_move_SM(self):
        '''
        state order: a, f, w, v
        sig_f: sigma f, the noise std on observation of firefly position
        sig_a:  ??? acceration
        xi_f: prior of firelfy position
        xi_v: prior of firelfy motion
        xs_w: prior of self motion

        return: covariance
        '''
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        sig_w = torch.exp(self.hsig_w)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_w = torch.exp(self.hxi_w)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        sig_a = torch.exp(self.hsig_a)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - time**2

        ratio = (sig_f/sig_w)**2

        det = ((Ct + t2_avg/(xi_f_**2))*(ratio + 1/(xi_v_**2) + 1/(xi_w_**2))
               + (1/(xi_v_**2))*(1 + 1/(xi_f_**2))*(ratio + 1/xi_w_**2))
        prefactor = 1/det

        c_ff = prefactor*((1/xi_v_**2 + Ct) *
                          (ratio + 1/xi_w_**2) + Ct/(xi_v_**2))
        c_fw = prefactor*(time/(xi_v_**2))*ratio
        c_fv = - prefactor*time*Ct*(ratio + 1/xi_w_**2 + 1/xi_v_**2)

        c_wf = - prefactor*(time/((xi_f_*xi_v_)**2))
        c_ww = prefactor*ratio * \
            ((1 + 1/xi_f_**2)/(xi_v_**2) + Ct + t2_avg/(xi_f_**2))
        c_wv = - prefactor*(Ct/(xi_v_**2))*(1 + 1/xi_f_**2)

        c_vf = prefactor*(time/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_w_**2))
        c_vw = prefactor*((sig_f/sig_w)**2)*(Ct + t2_avg/(xi_f_**2))
        c_vv = prefactor*Ct*(1 + 1/(xi_f_**2)) * \
            ((sig_f/sig_w)**2 + 1/(xi_w_**2))

        # ....

        Sig_ff = ((sig_f**2)/t_obs)*(c_ff**2 + (1/ratio)
                                     * (c_fw**2) + (1/Ct)*(c_fv**2))
        Sig_fw = ((sig_f**2)/t_obs)*(c_ff*c_wf + (1/ratio)
                                     * (c_fw*c_ww) + (1/Ct)*(c_fv*c_wv))
        Sig_fv = ((sig_f**2)/t_obs)*(c_ff*c_vf + (1/ratio)
                                     * (c_fw*c_vw) + (1/Ct)*(c_fv*c_vv))

        # Sig_wf = ((sig_f**2)/t_obs)*( c_ff*c_wf + (1/ratio)*(c_fw*c_ww) + (1/Ct)*(c_fv*c_wv)  )
        Sig_ww = ((sig_f**2)/t_obs)*(c_wf**2 + (1/ratio)
                                     * (c_ww**2) + (1/Ct)*(c_wv**2))
        Sig_wv = ((sig_f**2)/t_obs)*(c_vf*c_wf + (1/ratio)
                                     * (c_vw*c_ww) + (1/Ct)*(c_vv*c_wv))

        # Sig_vf = ((sig_f**2)/t_obs)*( c_ff*c_vf + (1/ratio)*(c_fw*c_vw) + (1/Ct)*(c_fv*c_vv)  )
        # Sig_vw = ((sig_f**2)/t_obs)*( c_vf*c_wf + (1/ratio)*(c_vw*c_ww) + (1/Ct)*(c_vv*c_wv)  )
        Sig_vv = ((sig_f**2)/t_obs)*(c_vf**2 + (1/ratio)
                                     * (c_vw**2) + (1/Ct)*(c_vv**2))

        Sig = torch.tensor([[sig_a**2, 0, 0, 0],
                            [0, Sig_ff, Sig_fw, Sig_fv],
                            [0, Sig_fw, Sig_ww, Sig_wv],
                            [0, Sig_fv, Sig_wv, Sig_vv]])
        return Sig

    def get_Sigma_inv_move_SM(self):
        Sig = 0
        return Sig

    def get_Sigma_stat_SM(self):
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        sig_w = torch.exp(self.hsig_w)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_w = torch.exp(self.hxi_w)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        sig_a = torch.exp(self.hsig_a)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - time**2

        # mu_f_ = (  ( ratio + 1/xi_w_**2 + Ct )*f_avg
        #          + (time)*ratio*w_avg
        #          - time*Ct*v_rel )

        # mu_w_ = (-  time/((xi_f_)**2) )*f_avg
        #          + ratio*( 1+ 1/xi_f_**2  )*w_avg
        #          - Ct*( 1 + 1/xi_f_**2 )*v_rel )

        # mu_v_ = (  (time/(xi_f_**2) )*( (sig_f/sig_w)**2 + 1/(xi_w_**2) )*f_avg
        #          + ( (sig_f/sig_w)**2 )*( Ct + t2_avg/(xi_f_**2) )*w_avg
        #          + Ct*(1 + 1/(xi_f_**2))*( (sig_f/sig_w)**2 + 1/(xi_w_**2) )*v_rel )

        ratio = (sig_f/sig_w)**2

        det_stat = Ct + t2_avg / \
            (xi_f_**2) + (1 + 1/(xi_f_**2))*((sig_f/sig_w)**2 + 1/xi_w_**2)
        prefactor_stat = 1/det_stat

        det_move = ((Ct + t2_avg/(xi_f_**2))*(ratio + 1/(xi_v_**2) + 1/(xi_w_**2))
                    + (1/(xi_v_**2))*(1 + 1/(xi_f_**2))*(ratio + 1/xi_w_**2))
        prefactor_move = 1/det_move

        c_ff = prefactor_stat*(ratio + 1/xi_w_**2 + Ct)
        c_fw = prefactor_stat*time*ratio
        c_fv = - prefactor_stat*time*Ct

        c_wf = - prefactor_stat*time/(xi_f_**2)
        c_ww = prefactor_stat*ratio*(1 + 1/xi_f_**2)
        c_wv = - prefactor_stat*Ct*(1 + 1/xi_f_**2)

        c_vf = prefactor_move*(time/(xi_f_**2)) * \
            ((sig_f/sig_w)**2 + 1/(xi_w_**2))
        c_vw = prefactor_move*((sig_f/sig_w)**2)*(Ct + t2_avg/(xi_f_**2))
        c_vv = prefactor_move*Ct*(1 + 1/(xi_f_**2)) * \
            ((sig_f/sig_w)**2 + 1/(xi_w_**2))

        # ....

        Sig_ff = ((sig_f**2)/t_obs)*(c_ff**2 + (1/ratio)
                                     * (c_fw**2) + (1/Ct)*(c_fv**2))
        Sig_fw = ((sig_f**2)/t_obs)*(c_ff*c_wf + (1/ratio)
                                     * (c_fw*c_ww) + (1/Ct)*(c_fv*c_wv))
        Sig_fv = ((sig_f**2)/t_obs)*(c_ff*c_vf + (1/ratio)
                                     * (c_fw*c_vw) + (1/Ct)*(c_fv*c_vv))

        Sig_ww = ((sig_f**2)/t_obs)*(c_wf**2 + (1/ratio)
                                     * (c_ww**2) + (1/Ct)*(c_wv**2))
        Sig_wv = ((sig_f**2)/t_obs)*(c_vf*c_wf + (1/ratio)
                                     * (c_vw*c_ww) + (1/Ct)*(c_vv*c_wv))
        Sig_vv = ((sig_f**2)/t_obs)*(c_vf**2 + (1/ratio)
                                     * (c_vw**2) + (1/Ct)*(c_vv**2))

        Sig = torch.tensor([[sig_a**2, 0, 0, 0],
                            [0, Sig_ff, Sig_fw, Sig_fv],
                            [0, Sig_fw, Sig_ww, Sig_wv],
                            [0, Sig_fv, Sig_wv, Sig_vv]])

        return Sig

    def get_Sigma_inv_stat_SM(self):
        Sig = 0
        return Sig

    # seems ok
    def get_W_move_SM(self):
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        sig_w = torch.exp(self.hsig_w)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_w = torch.exp(self.hxi_w)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        W_aa = torch.exp(self.hW_aa)

        time = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - time**2

        ratio = (sig_f/sig_w)**2

        W_ff = (1/xi_v_**2 + Ct)*(ratio + 1/xi_w_**2) + Ct/(xi_v_**2)
        W_fw = - time/((xi_v_*xi_w_)**2)
        W_fv = (time/(xi_v_**2))*(ratio + 1/xi_w_**2)

        W_wf = - time/((xi_f_*xi_v_)**2)
        W_ww = (ratio/(xi_v_**2))*(1 + 1/xi_f_**2) + \
            (ratio + 1/xi_v_**2)*(Ct + t2_avg/(xi_f_**2))
        W_wv = - (1/xi_v_**2)*(Ct + t2_avg/(xi_f_**2))

        W_vf = (time/(xi_f_**2))*(ratio + 1/xi_w_**2)
        W_vw = - (1/xi_w_**2)*(Ct + t2_avg/(xi_f_**2))
        W_vv = (Ct + t2_avg/(xi_f_**2))*(ratio + 1/xi_w_**2)

        det = ((Ct + t2_avg/(xi_f_**2))*(ratio + 1/(xi_v_**2) + 1/(xi_w_**2))
               + (1/(xi_v_**2))*(1 + 1/(xi_f_**2))*(ratio + 1/xi_w_**2))
        prefactor = 1/det

        W_move = prefactor*torch.tensor([[0, 0, 0, 0],
                                         [0, W_ff, W_fw, W_fv],
                                         [0, W_wf, W_ww, W_wv],
                                         [0, W_vf, W_vw, W_vv]])
        W_move[0, 0] = W_aa
        return W_move

    def get_W_stat_SM(self):
        t_obs = self.t_obs
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_w = torch.exp(self.hxi_w)
        sig_f = torch.exp(self.hsig_f)
        sig_w = torch.exp(self.hsig_w)

        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        time = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - time**2
        ratio = (sig_f/sig_w)**2

        det_stat = Ct + t2_avg / \
            (xi_f_**2) + (1 + 1/(xi_f_**2))*((sig_f/sig_w)**2 + 1/xi_w_**2)
        prefactor_stat = 1/det_stat
        W_ff = prefactor_stat*(Ct + ratio + 1/xi_w_**2)
        W_fw = prefactor_stat*(- time/(xi_w_**2))
        W_fv = prefactor_stat*(ratio + 1/xi_w_**2)

        W_wf = prefactor_stat*(-time/(xi_f_**2))
        W_ww = prefactor_stat*(Ct + t2_avg/(xi_f_**2) + ratio*(1 + 1/xi_f_**2))
        W_wv = prefactor_stat*(- (Ct + t2_avg/(xi_f_**2)))

        det_move = ((Ct + t2_avg/(xi_f_**2))*(ratio + 1/(xi_v_**2) + 1/(xi_w_**2))
                    + (1/(xi_v_**2))*(1 + 1/(xi_f_**2))*(ratio + 1/xi_w_**2))
        prefactor_move = 1/det_move

        W_vf = prefactor_move*(time/(xi_f_**2))*(ratio + 1/xi_w_**2)
        W_vw = - prefactor_move*(1/xi_w_**2)*(Ct + t2_avg/(xi_f_**2))
        W_vv = prefactor_move*(Ct + t2_avg/(xi_f_**2))*(ratio + 1/xi_w_**2)

        W_stat = torch.tensor([[W_ff, W_fw, W_fv],
                               [W_wf, W_ww, W_wv],
                               [W_vf, W_vw, W_vv]])
        return W_stat

    def get_det_Sigma_move_SM():
        return 0

    def get_det_Sigma_stat_SM():
        return 0

    def get_det_ratio_SM(self):
        t_obs = self.t_obs
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_w = torch.exp(self.hxi_w)
        sig_f = torch.exp(self.hsig_f)
        sig_w = torch.exp(self.hsig_w)

        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        time = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - time**2

        det = ((Ct + t2_avg/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_v_**2) + 1/(xi_w_**2))
               + (1/(xi_v_**2))*(1 + 1/(xi_f_**2))*((sig_f/sig_w)**2 + 1/xi_w_**2))

        det_red = Ct + t2_avg/(xi_f_**2) + (1 + 1/(xi_f_**2)) * \
            ((sig_f/sig_w)**2 + 1/xi_w_**2)

        det_ratio = det/det_red
        return det_ratio

    def get_threshold_SM(self):
        '''
        t_obs: total obs steps
        xi_v: prior of 
        '''
        t_obs = self.t_obs
        xi_v = torch.exp(self.hxi_v)
        sig_f = torch.exp(self.hsig_f)
        delta0 = self.delta0

        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        extra_factor = t_obs/(sig_f**2)

        det_ratio = self.get_det_ratio_SM()

        thresh_sq = (torch.log((xi_v_**2)*det_ratio) -
                     2*delta0)/(extra_factor*det_ratio)
        q = torch.sqrt(nn.functional.relu(thresh_sq))
        return q

    # pretty sure this is redundant, since eqns should not change
    def get_w_xv_A_SM(self, t, t_obs, t_wait, T):
        alpha = torch.exp(self.halpha)

        denom = 1/(alpha + T - t)
        w_tx = -1/denom
        w_tv = (t_obs + t_wait + T)/denom
        return w_tx, w_tv

    # shape = (2, num_t_pts)
    def get_w_o_move_A_SM(self, t, t_obs, t_wait, T):
        alpha = torch.exp(self.halpha)

        denom = 1/(alpha + T - t)
        w_f = 1/denom
        w_w = - t_obs/denom
        w_a = 0.5*(t_wait**2 + T ^ 2)/denom

        w_o = torch.stack((w_a, w_f, w_w))
        return w_o

    # shape = (num_t_pts)
    def get_w_o_stat_A_SM(self, t, t_obs, t_wait, T):
        alpha = torch.exp(self.halpha)

        denom = 1/(alpha + T - t)
        w_f = 1/denom
        w_w = -t_obs/denom

        w_o = torch.stack((w_f, w_w))
        return w_o
