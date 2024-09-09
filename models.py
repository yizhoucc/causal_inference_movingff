# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:27:38 2024

@author: johnv
"""

import numpy as np
import torch
import torch.nn as nn


class Model_NoSM_ControlA(nn.Module):

    def __init__(self, t_obs, T_control):
        '''tobs, how long the obs phase last.
        t control, '''
        super().__init__()
        # h stands for logged to be all positive
        # Obs model parameters
        self.hsig_f = nn.Parameter(torch.rand(1)) # obj motion obs noise
        self.hsig_w = nn.Parameter(torch.rand(1))  # self motion obs noise

        self.hxi_f = nn.Parameter(torch.rand(1)) # inital distance
        self.hxi_w = nn.Parameter(torch.rand(1)) # self motion
        self.hxi_v = nn.Parameter(torch.rand(1)) # obj motion

        # not sure if I need to exp this
        self.delta0 = nn.Parameter(torch.rand(1)) # move or not, decision prior
        # -------------------------
        # Control model parameters
        self.halpha = nn.Parameter(torch.rand(1)) # acc?
        self.hg = nn.Parameter(torch.rand(1)) # control gain

        # Other parameters
        self.hW_aa = nn.Parameter(torch.rand(1))
        self.hsig_a = nn.Parameter(torch.rand(1))

        self.t_obs = t_obs # number of steps in obs phase
        self.T_control = T_control # totoal time: obs+control

    def forward(self, u_t, x_t, z, t, t_wait, u_t_SM, x_t_SM, z_SM, t_SM, t_wait_SM):
        # z: a, f_0, v, shape = (num_trials, 3) this is belief??
        # t, shape = (num_trials, num_t_pts)
        # t_wait, shape = (num_trials, num_t_pts)

        # Process input
        # z_noa = z[:,1:]
        # f0 = z[:,0]; w = z[:,1]; v = z[:,2]; a = z[:,3]
        dt = (t[:, 1] - t[:, 0])                    # shape = (num_trials,None)
        T = (self.T_control - t_wait)[:, None]     # shape = (num_trials,None)

        # SM stands for self motion
        dt_SM = (t_SM[:, 1] - t[:, 0])
        T_SM = (self.T_control - t_wait_SM)[:, None]

        # num_trials, num_t_pts = f0.shape

        # Get model parameters
        t_obs = self.t_obs
        # sig_f = torch.exp(self.hsig_f); sig_w = torch.exp(self.hsig_w)
        # xi_f = torch.exp(self.hxi_f); xi_w = torch.exp(self.hxi_w); xi_v = torch.exp(self.hxi_v)
        # delta0 = self.delta0
        # alpha = torch.exp(self.halpha)
        # g = torch.exp(self.hg)
        # sig_a = torch.exp(self.hsig_a)

        # Get relevant obs model quantities

        # NO SELF-MOTION ---------------------------------------------------------
        # Sigma_move = self.get_Sigma_move_noSM()
        # Sigma_stat = self.get_Sigma_stat_noSM()#

        # getting the covs
        # Get inverse average covariance matrix, ff move model
        Sigma_inv_move = self.get_Sigma_inv_move_noSM()
        # Get inverse average covariance matrix, ff stat model
        Sigma_inv_stat = self.get_Sigma_inv_stat_noSM()
        # Get average posterior mean z -> z-hat matrix, move model
        W_move = self.get_W_move_noSM()
        # Get average posterior mean z -> z-hat matrix, stat model
        W_stat = self.get_W_stat_noSM()
        det_Sig_move = self.get_det_Sig_move_noSM()
        det_Sig_stat = self.get_det_Sig_stat_noSM()
        # Get decision threshold (for causal inference about ff moving for not)
        q = self.get_threshold_noSM()



        # SELF-MOTION ---------------------------------------------------------
        Sigma_move_SM = self.get_Sigma_move_SM()
        Sigma_stat_SM = self.get_Sigma_stat_SM()
        W_move_SM = self.get_W_move_SM()
        W_stat_SM = self.get_W_stat_SM()
        Sigma_inv_move_SM = torch.inverse(Sigma_move_SM)  # COMPUTE
        Sigma_inv_stat_SM = torch.inverse(Sigma_stat_SM)
        det_Sig_move_SM = torch.det(Sigma_move_SM)
        det_Sig_stat_SM = torch.det(Sigma_stat_SM)

        # could be using the same q. threshold
        q_SM = self.get_threshold_SM()

        # Get relevant control model quantities
        # vector of weights on u_t^*, ignoring v, ignoring time-dep prefactor
        w_o_move = self.get_w_o_move_A_noSM(t, t_obs, t_wait, T)
        w_o_stat = self.get_w_o_stat_A_noSM(t, t_obs, t_wait, T)
        w_tx, w_tv = self.get_w_tv_A_noSM(t, t_obs, t_wait, T)

        w_o_move_SM = self.get_w_o_move_A_SM(t_SM, t_obs, t_wait_SM, T_SM)
        w_o_stat_SM = self.get_w_o_stat_A_SM(t_SM, t_obs, t_wait_SM, T_SM)
        w_tx_SM, w_tv_SM = self.get_w_tv_A_SM(t_SM, t_obs, t_wait_SM, T_SM)

        # getting log likelihood
        L_noSM = self.get_L(u_t, x_t, z, dt, Sigma_inv_move, Sigma_inv_stat, det_Sig_move, det_Sig_stat,
                            W_move, W_stat, q, w_o_move, w_o_stat, w_tx, w_tv)
        L_SM = self.get_L(u_t_SM, x_t_SM, z_SM, dt_SM, Sigma_inv_move_SM, Sigma_inv_stat_SM, det_Sig_move_SM, det_Sig_stat_SM,
                          W_move_SM, W_stat_SM, q_SM, w_o_move_SM, w_o_stat_SM, w_tx_SM, w_tv_SM)

        L = L_noSM + L_SM
        return L

    def get_L(self, u_t, x_t, z, dt, Sigma_inv_move, Sigma_inv_stat, det_Sig_move, det_Sig_stat, W_move, W_stat, qqq, w_o_move, w_o_stat, w_tx, w_tv):
        '''
        z: latent variables. belief? 
        u, controls
        x, states
        Sigmas:
        W matrix: the M matrixs in notes
        q: decision threshold
        w_o:
        w_t:
        
        '''
        z_noa = z[:, 1:]
        num_t_pts = u_t.shape[1]

        # sig_f = torch.exp(self.hsig_f); sig_w = torch.exp(self.hsig_w)
        # xi_f = torch.exp(self.hxi_f); xi_w = torch.exp(self.hxi_w); xi_v = torch.exp(self.hxi_v)
        # delta0 = self.delta0
        # alpha = torch.exp(self.halpha)
        g = torch.exp(self.hg)
        sig_a = torch.exp(self.hsig_a)

        # Transform matrices
        Sig_inv_move_vv = Sigma_inv_move[-1, -1]
        Sig_inv_stat_vv = Sigma_inv_stat[-1, -1]
        Sig_inv_move_vo = Sigma_inv_move[-1, :-1]
        Sig_inv_stat_vo = Sigma_inv_stat[-1, :-1]
        Sig_inv_move_oo = Sigma_inv_move[:-1, :-1]
        Sig_inv_stat_oo = Sigma_inv_stat[:-1, :-1]

        W_move_v = W_move[-1, :]
        W_stat_v = W_stat[-1, :]
        W_move_o = W_move[:-1, :]
        W_stat_o = W_stat[:-1, :]

        # A, 7.19.
        A_move = Sig_inv_move_oo[None, :, :] + (dt[:, None, None]/g**2)*(
            w_o_move @ w_o_move.T)[None, :, :]    # shape = (num_trials, )
        A_stat = Sig_inv_stat_oo[None, :, :] + \
            (dt[:, None, None]/g**2)*(w_o_stat @ w_o_stat.T)[None, :, :]
        
        A_inv_move = torch.inverse(A_move)
        det_A_move = torch.det(A_move)
        A_inv_stat = torch.inverse(A_stat)
        det_A_stat = torch.det(A_stat)

        # J, 7.21
        J_r_move = (Sig_inv_move_oo @ W_move_o @ z + Sig_inv_move_vo @ W_move_v @ z
                    + (dt[:, None]/g**2)*torch.sum(w_o_move[None, :, :] *
                                                   (u_t[:, None, :] - w_tx[None, None, :]*x_t[:, None, :]), axis=-1)
                    ) # residual along with jv*v
        J_v_move = Sig_inv_move_vo[None, :, :] + \
            (dt[:, None, None]/g**2)*(w_o_move @ w_tv)[None, :, :] # jv * v
        J_r_stat = (Sig_inv_stat_oo @  W_stat_o @ z_noa + Sig_inv_stat_vo @ W_stat_v @ z_noa
                    + (dt[:, None]/g**2)*torch.sum(w_o_stat[None, :, :]*(u_t[:, None, :] - w_tx[None, None, :]*x_t[:, None, :]), axis=-1))
        J_v_stat = Sig_inv_stat_vo[None, :, :]
        
        J_move = (Sig_inv_move_vv*(W_move_v @ z) + torch.einsum('ij,jk,bk->bi', Sig_inv_move_vo, W_move_v, z)
                  + (dt/g**2)*torch.sum(w_tv[None, :]
                                        * (u_t - w_tx[None, :]*x_t), axis=-1)
                  - torch.einsum('bi,bij,bj->b', J_r_move, A_inv_move, J_v_move))
        k_move = Sig_inv_move_vv + (1/g**2)*torch.sum((w_tv**2)*dt) - \
            torch.einsum('bi,bij,bj->b', J_v_move, A_inv_move, J_v_move)
        Z_move = (0.5*torch.einsum('bi,ij,jk,kl,bl->b', z, W_move, Sigma_inv_move, W_move, z)
                  + 0.5*(1/g**2)*torch.sum(dt[:, None] *
                                           ((u_t - w_tx[None, :]*x_t)**2), axis=-1)
                  - 0.5*torch.einsum('bi,bij,bj->b', J_r_move,
                                     A_inv_move, J_r_move)
                  - 0.5*J_move*J_move/k_move
                  )
        det_prefactor_move = k_move*det_A_move*det_Sig_move*(sig_a**2)

        # 7.27
        J_stat = (Sig_inv_stat_vv*(W_stat_v @ z_noa) + torch.einsum('ij,jk,bk->bi', Sig_inv_stat_vo, W_stat_v, z_noa)
                  - torch.einsum('bi,bij,bj->b', J_r_stat, A_inv_stat, J_v_stat))
        k_stat = Sig_inv_stat_vv - \
            torch.einsum('bi,bij,bj->b', J_v_stat, A_inv_stat, J_v_stat)
        det_prefactor_stat = k_stat*det_A_stat*det_Sig_stat
        Z_stat = (0.5*torch.einsum('bi,ij,jk,kl,bl->b', z_noa, W_stat, Sigma_inv_stat, W_stat, z_noa)
                  + 0.5*(1/g**2)*torch.sum(dt[:, None] *
                                           ((u_t - w_tx[None, :]*x_t)**2), axis=-1)
                  - 0.5*torch.einsum('bi,bij,bj->b', J_r_stat,
                                     A_inv_stat, J_r_stat)
                  - 0.5*J_stat*J_stat/k_stat
                  )

        # Final objective, L_move and L_stat each have shape (num_trials)
        erf_part_move = 1 + 0.5*torch.erf((J_move - k_move*qqq)/(torch.sqrt(
            2*k_move))) - 0.5*torch.erf((J_move + k_move*qqq)/(torch.sqrt(2*k_move)))
        L_move = (1/torch.sqrt(det_prefactor_move)) * \
            torch.exp(-Z_move)*erf_part_move  # 7.26

        erf_part_stat = - 0.5*torch.erf((J_stat - k_stat*qqq)/(torch.sqrt(
            2*k_stat))) + 0.5*torch.erf((J_stat + k_stat*qqq)/(torch.sqrt(2*k_stat))) 
        
        L_stat = (1/torch.sqrt(det_prefactor_stat)) * \
            torch.exp(-Z_stat)*erf_part_stat # 7.26

        L = torch.sum(num_t_pts*torch.log((2*np.pi)*g*g/dt) -
                      torch.log(L_move + L_stat)) # eqation 7.28
        return L

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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (t_avg**2)/(xi_f_**2)
        prefactor = ((sig_f**2)/t_obs)*(1/(det_inv**2))

        Sig_ff = (C_t + (1/xi_v_**2))**2 + C_t*(t_avg**2)
        Sig_fv = t_avg*(1/((xi_f_*xi_v_)**2) - C_t)
        Sig_vv = (t_avg/(xi_f_**2))**2 + C_t*((1 + 1/xi_f_**2)**2)

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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

        prefactor = (t_obs/(sig_f**2))*(1/C_t)
        Sig_ff = ((t_avg)/(xi_f_**2))**2 + C_t*((1 + (1/xi_f_)**2)**2)
        Sig_fv = - t_avg*(1/((xi_f_*xi_v_)**2) - C_t)
        Sig_vv = (C_t + (1/xi_v_**2))**2 + C_t*(t_avg**2)

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
        '''2x2. f, v'''
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (t_avg**2)/(xi_f_**2)

        prefactor = (sig_f**2)/t_obs
        Sig_ff = 1/((1 + (1/xi_f_**2))**2)
        Sig_fv = (1/det_inv)*(t_avg/(xi_f_**2 + 1))
        Sig_vv = (1/det_inv**2)*(((t_avg)/(xi_f_**2))
                                 ** 2 + C_t*((1 + (1/xi_f_)**2)**2))

        Sig = torch.tensor([[Sig_ff, Sig_fv], [Sig_fv, Sig_vv]])

        result = prefactor*Sig
        return result

    # checked; shape = 2x2
    def get_Sigma_inv_stat_noSM(self):
        '''2x2. f v'''
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (t_avg**2)/(xi_f_**2)

        prefactor = (t_obs/(sig_f**2))*(1/C_t)
        Sig_ff = ((t_avg)/(xi_f_**2))**2 + C_t*((1 + (1/xi_f_)**2)**2)
        Sig_fv = - det_inv*(t_avg/(xi_f_**2 + 1))
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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (t_avg**2)/(xi_f_**2)

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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (t_avg**2)/(xi_f_**2)

        prefactor = ((sig_f**2)/t_obs)**2
        result = prefactor*C_t/(det_inv**2)
        return result

    # Average posterior mean
    # checked
    def get_W_move_noSM(self):
        '''a, f, v'''
        t_obs = self.t_obs
        sig_f = torch.exp(self.hsig_f)
        xi_f = torch.exp(self.hxi_f)
        xi_v = torch.exp(self.hxi_v)
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        W_aa = torch.exp(self.hW_aa)

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (t_avg**2)/(xi_f_**2)
        prefactor = 1/det_inv

        W_ff = C_t + 1/xi_v_**2
        W_fv = t_avg/(xi_v_**2)
        W_vf = t_avg/(xi_f_**2)
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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        C_t = t2_avg - t_avg**2

        det_inv = (1 + (1/xi_f_**2))*(C_t + (1/xi_v_**2)) + \
            (t_avg**2)/(xi_f_**2)

        W_ff = (1/(1 + 1/xi_f_**2))
        W_fv = (t_avg/(1 + 1/xi_f_**2))
        W_vf = (1/det_inv)*(t_avg/(xi_f_**2))
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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2

        ratio = (sig_f/sig_w)**2

        det = ((Ct + t2_avg/(xi_f_**2))*(ratio + 1/(xi_v_**2) + 1/(xi_w_**2))
               + (1/(xi_v_**2))*(1 + 1/(xi_f_**2))*(ratio + 1/xi_w_**2))
        prefactor = 1/det

        c_ff = prefactor*((1/xi_v_**2 + Ct) *
                          (ratio + 1/xi_w_**2) + Ct/(xi_v_**2))
        c_fw = prefactor*(t_avg/(xi_v_**2))*ratio
        c_fv = - prefactor*t_avg*Ct*(ratio + 1/xi_w_**2 + 1/xi_v_**2)

        c_wf = - prefactor*(t_avg/((xi_f_*xi_v_)**2))
        c_ww = prefactor*ratio * \
            ((1 + 1/xi_f_**2)/(xi_v_**2) + Ct + t2_avg/(xi_f_**2))
        c_wv = - prefactor*(Ct/(xi_v_**2))*(1 + 1/xi_f_**2)

        c_vf = prefactor*(t_avg/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_w_**2))
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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2

        # mu_f_ = (  ( ratio + 1/xi_w_**2 + Ct )*f_avg
        #          + (t_avg)*ratio*w_avg
        #          - t_avg*Ct*v_rel )

        # mu_w_ = (-  t_avg/((xi_f_)**2) )*f_avg
        #          + ratio*( 1+ 1/xi_f_**2  )*w_avg
        #          - Ct*( 1 + 1/xi_f_**2 )*v_rel )

        # mu_v_ = (  (t_avg/(xi_f_**2) )*( (sig_f/sig_w)**2 + 1/(xi_w_**2) )*f_avg
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
        c_fw = prefactor_stat*t_avg*ratio
        c_fv = - prefactor_stat*t_avg*Ct

        c_wf = - prefactor_stat*t_avg/(xi_f_**2)
        c_ww = prefactor_stat*ratio*(1 + 1/xi_f_**2)
        c_wv = - prefactor_stat*Ct*(1 + 1/xi_f_**2)

        c_vf = prefactor_move*(t_avg/(xi_f_**2)) * \
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

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2

        ratio = (sig_f/sig_w)**2

        W_ff = (1/xi_v_**2 + Ct)*(ratio + 1/xi_w_**2) + Ct/(xi_v_**2)
        W_fw = - t_avg/((xi_v_*xi_w_)**2)
        W_fv = (t_avg/(xi_v_**2))*(ratio + 1/xi_w_**2)

        W_wf = - t_avg/((xi_f_*xi_v_)**2)
        W_ww = (ratio/(xi_v_**2))*(1 + 1/xi_f_**2) + \
            (ratio + 1/xi_v_**2)*(Ct + t2_avg/(xi_f_**2))
        W_wv = - (1/xi_v_**2)*(Ct + t2_avg/(xi_f_**2))

        W_vf = (t_avg/(xi_f_**2))*(ratio + 1/xi_w_**2)
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
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2
        ratio = (sig_f/sig_w)**2

        det_stat = Ct + t2_avg / \
            (xi_f_**2) + (1 + 1/(xi_f_**2))*((sig_f/sig_w)**2 + 1/xi_w_**2)
        prefactor_stat = 1/det_stat
        W_ff = prefactor_stat*(Ct + ratio + 1/xi_w_**2)
        W_fw = prefactor_stat*(- t_avg/(xi_w_**2))
        W_fv = prefactor_stat*(ratio + 1/xi_w_**2)

        W_wf = prefactor_stat*(-t_avg/(xi_f_**2))
        W_ww = prefactor_stat*(Ct + t2_avg/(xi_f_**2) + ratio*(1 + 1/xi_f_**2))
        W_wv = prefactor_stat*(- (Ct + t2_avg/(xi_f_**2)))

        det_move = ((Ct + t2_avg/(xi_f_**2))*(ratio + 1/(xi_v_**2) + 1/(xi_w_**2))
                    + (1/(xi_v_**2))*(1 + 1/(xi_f_**2))*(ratio + 1/xi_w_**2))
        prefactor_move = 1/det_move

        W_vf = prefactor_move*(t_avg/(xi_f_**2))*(ratio + 1/xi_w_**2)
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
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2

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
                     2*delta0)/(extra_factor*det_ratio) #4.32
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
