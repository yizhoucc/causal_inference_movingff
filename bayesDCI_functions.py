# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:16:27 2024

@author: johnv
"""

import numpy as np
from scipy.special import erf


class BayesDCIStatic():
    def __init__(self, params):

        self.sigma_obs, self.sigma_obs_inv = self.construct_sigma_obs(params)
        self.sigma_0 = self.construct_sigma_0(params)
        self.params = params

    def construct_sigma_obs(self, params):
        return 0, 0

    def construct_sigma_0(self, params):
        return 0

    def get_posterior(self, y, t_obs):
        pass

    def generate_obs(self, num_obs):
        x = 0
        return x

    def get_suff_stats(self, x):
        y = 0
        return y

# =====================================================================

# 1 variable model with one latent, w
# class implements bayesian algorithm to decide whether w = 0 or not


class BayesDCIStatic_SMonly(BayesDCIStatic):

    def get_posterior(self, y, t_obs):

        xi_w, sig_w = self.params['xi_w'], self.params['sig_w']
        w_avg = y

        cov_inv = 1/(xi_w**2) + t_obs/(sig_w**2)
        det = cov_inv
        cov = 1/cov_inv
        mu = ((t_obs/(sig_w**2))/cov_inv)*w_avg

        return mu, cov, cov_inv, det

    def get_det_cov_red(self, t_obs):
        return 1.

    def generate_obs(self, z, t, num_runs=1):
        # t specifies which time points get an observation

        sig_w = self.params['sig_w']
        w = z
        dt = (np.roll(t, -1) - t)[:-1]

        x_t = np.random.normal(loc=w[None, None, :], scale=sig_w/np.sqrt(
            dt[:, None, None]), size=(len(dt), num_runs, *w.shape))
        return x_t

    def get_suff_stats(self, x_t):
        w_t = x_t

        w_avg = np.mean(w_t, axis=0)
        return w_avg

    def get_det_ratio(self, t_obs):
        xi_w, sig_w = self.params['xi_w'], self.params['sig_w']

        det_ratio = 1/(xi_w**2) + t_obs/(sig_w**2)
        return det_ratio

    def get_decision(self, y, t_obs):
        # decide whether model should be reduced (delta >= 0) or not
        # return decision (1 or 0) and associated confidence (0 <= p <= 1)

        # required params
        delta_0, xi_w = self.params['delta_0'], self.params['xi_w']
        det_ratio = self.get_det_ratio(t_obs)
        mu, _, _, _ = self.get_posterior(y, t_obs)

        delta = 0.5*det_ratio*(mu**2) - 0.5 * \
            np.log((xi_w**2)*det_ratio) + delta_0

        delta_abs = np.abs(delta)     # get decision confidence
        p_correct = np.exp(delta_abs)/(1 + np.exp(delta_abs))

        decision = (delta >= 0).astype(int)
        return decision, p_correct

    def get_decision_boundary(self, t_obs):

        # required params
        delta_0, xi_w = self.params['delta_0'], self.params['xi_w']
        det_ratio = self.get_det_ratio(t_obs)

        thresh_sq = (np.log((xi_w**2)*det_ratio) - 2*delta_0)/det_ratio
        q = np.sqrt(np.maximum(thresh_sq, 0))
        return q

    def get_avg_mu_sig(self, z, t_obs):
        xi_w, sig_w = self.params['xi_w'], self.params['sig_w']
        w = z

        cov_inv = 1/(xi_w**2) + t_obs/(sig_w**2)
        mu = ((t_obs/(sig_w**2))/cov_inv)*w
        sig = (np.sqrt(t_obs)/sig_w)/cov_inv
        return mu, sig

    def get_psycho_curve(self, z, t_obs):
        mu, sig = self.get_avg_mu_sig(z, t_obs)
        q = self.get_decision_boundary(t_obs)

        curve = 0.5*(erf((q - mu)/(np.sqrt(2)*sig)) +
                     erf((q + mu)/(np.sqrt(2)*sig)))
        return curve


# =====================================================================

# 1D model with two latents (f0, v)
# class implements bayesian algorithm to decide whether v = 0 or not
class BayesDCIStatic_noSM(BayesDCIStatic):

    def get_posterior(self, y, t_obs):

        pass

        return 0

    # has been checked
    def get_post_v(self, y, t_obs):
        xi_f = self.params['xi_f']
        sig_f = self.params['sig_f']

        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2
        f_avg, v_hat = y

        det = self.get_det_cov(t_obs)
        prefactor = 1/det

        mu_v = prefactor*((t_avg/(xi_f_**2))*f_avg +
                          Ct*(1 + 1/(xi_f_**2))*v_hat)
        return mu_v

    def generate_obs(self, z, t, num_runs=1):
        sig_f = self.params['sig_f']
        f, v = z
        dt = (np.roll(t, -1) - t)[:-1]
        f_t_avg = f[None, None, :] + v[None, None, :]*t[:, None, None]

        f_t = np.random.normal(
            loc=f_t_avg, scale=sig_f/np.sqrt(dt[:, None, None]), size=(len(dt), num_runs, *f.shape))
        return f_t

    def get_suff_stats(self, x_t, t_obs):
        f_t = x_t

        T = f_t.shape[0]
        t_ish = np.arange(1, T+1)/T

        f_avg = np.mean(f_t, axis=0)
        f_avg_ = f_avg[None, :]
        v_hat = (12/t_obs)*np.mean((f_t - f_avg_)*(t_ish - 0.5), axis=0)
        y = f_avg, v_hat
        return y

    # has been checked
    def get_det_cov(self, t_obs):
        xi_f, xi_v = self.params['xi_f'], self.params['xi_v']
        sig_f = self.params['sig_f']

        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2

        det = (1 + 1/(xi_f_**2))*(Ct + 1/(xi_v_**2)) + (t_avg**2)/(xi_f_**2)
        return det

    # has been checked; could also use simpler direct formula
    def get_det_ratio(self, t_obs):
        xi_f, sig_f = self.params['xi_f'], self.params['sig_f']
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)

        det = self.get_det_cov(self, t_obs)
        det_red = 1 + 1/(xi_f_**2)

        det_ratio = det/det_red
        return det_ratio

    # has been checked
    def get_decision(self, y, t_obs):
        # decide whether model should be reduced (delta >= 0) or not
        # return decision (1 or 0) and associated confidence (0 <= p <= 1)

        # required params
        delta_0, xi_v = self.params['delta_0'], self.params['xi_v']
        sig_f = self.params['sig_f']
        det_ratio = self.get_det_ratio(t_obs)
        mu = self.get_post_v(self, y, t_obs)
        extra_factor = t_obs/(sig_f**2)

        delta = 0.5*extra_factor*det_ratio * \
            (mu**2) - 0.5*np.log((xi_v_**2)*det_ratio) + delta_0

        delta_abs = np.abs(delta)     # get decision confidence
        p_correct = np.exp(delta_abs)/(1 + np.exp(delta_abs))

        decision = (delta >= 0).astype(int)
        return decision, p_correct

    # has been checked
    def get_decision_boundary(self, t_obs):
        # required params
        delta_0, xi_v = self.params['delta_0'], self.params['xi_v']
        sig_f = self.params['sig_f']
        det_ratio = self.get_det_ratio(t_obs)
        extra_factor = t_obs/(sig_f**2)

        thresh_sq = (np.log((xi_v_**2)*det_ratio) - 2*delta_0) / \
            (extra_factor*det_ratio)
        q = np.sqrt(np.maximum(thresh_sq, 0))
        return q

    # has been checked
    def get_avg_mu_sig(self, z, t_obs):
        xi_f, sig_f = self.params['xi_f'], self.params['sig_f']
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2
        f0, v = z

        prefactor = 1/self.get_det_cov(t_obs)
        mu = prefactor*((t_avg/(xi_f_**2))*f0 + (Ct + t2_avg/(xi_f_**2))*v)
        sig = prefactor*(sig_f/np.sqrt(t_obs))*np.sqrt((t_avg /
                                                        (xi_f_**2))**2 + Ct*((1 + 1/(xi_f_**2))**2))
        return mu, sig

    # has been checked
    def get_psycho_curve(self, z, t_obs):
        mu, sig = self.get_avg_mu_sig(z, t_obs)
        q = self.get_decision_boundary(t_obs)

        curve = 0.5*(erf((q - mu)/(np.sqrt(2)*sig)) +
                     erf((q + mu)/(np.sqrt(2)*sig)))
        return curve


# =====================================================================

# 1D model with three latents (f0, w, v)
# class implements bayesian algorithm to decide whether v = 0 or not
class BayesDCIStatic_full(BayesDCIStatic):
    # has been checked
    def get_posterior(self, y, t_obs):
        xi_f, xi_w, xi_v = self.params['xi_f'], self.params['xi_w'], self.params['xi_v']
        sig_f, sig_w = self.params['sig_f'], self.params['sig_w']

        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)

        f_avg, w_avg, v_rel = y
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2
        det = self.get_det_cov(t_obs)
        prefactor = 1/det

        ratio = (sig_f/sig_w)**2

        # Posterior mean ----------
        mu_f_ = (((1/xi_v_**2 + Ct)*(ratio + 1/xi_w_**2) + Ct/(xi_v_**2))*f_avg
                 + (t_avg/(xi_v_**2))*ratio*w_avg
                 - t_avg*Ct*(ratio + 1/xi_w_**2 + 1/xi_v_**2)*v_rel)

        mu_w_ = (- (t_avg/((xi_f_*xi_v_)**2))*f_avg
                 + ratio*((1 + 1/xi_f_**2)/(xi_v_**2) +
                          Ct + t2_avg/(xi_f_**2))*w_avg
                 - (Ct/(xi_v_**2))*(1 + 1/xi_f_**2)*v_rel)

        mu_v_ = ((t_avg/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_w_**2))*f_avg
                 + ((sig_f/sig_w)**2)*(Ct + t2_avg/(xi_f_**2))*w_avg
                 + Ct*(1 + 1/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_w_**2))*v_rel)
        mu = prefactor*np.array([mu_f_, mu_w_, mu_v_])
        # ------------------------------
        # Posterior covariance
        prefactor_overall = prefactor*(sig_f**2)/t_obs
        cov_ff = (1/xi_v_**2 + t2_avg)*(ratio + 1/xi_w_**2) + t2_avg/(xi_v_**2)
        cov_fw = t_avg/(xi_v_**2)
        cov_fv = - t_avg*(ratio + 1/xi_w_**2)
        cov_ww = (1 + 1/xi_f_**2)/(xi_v_**2) + Ct + t2_avg/(xi_f_**2)
        cov_wv = Ct + t2_avg/(xi_f_**2)
        cov_vv = Ct + t2_avg/(xi_f_**2) + (1 + 1/xi_f_**2)*(ratio + 1/xi_w_**2)

        cov = np.array([[cov_ff, cov_fw, cov_fv],
                        [cov_fw, cov_ww, cov_wv],
                        [cov_fv, cov_wv, cov_vv]])
        cov = prefactor_overall*cov
        return mu, cov

    # has been checked
    def get_post_v(self, y, t_obs):
        xi_f, xi_w = self.params['xi_f'], self.params['xi_w']
        sig_f, sig_w = self.params['sig_f'], self.params['sig_w']
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)

        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2
        f_avg, w_avg, v_rel = y

        det = self.get_det_cov(t_obs)

        prefactor = 1/det
        mu_v = prefactor*((t_avg/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_w_**2))*f_avg
                          + ((sig_f/sig_w)**2)*(Ct + t2_avg/(xi_f_**2))*w_avg
                          + Ct*(1 + 1/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_w_**2))*v_rel)
        return mu_v

    def generate_obs(self, z, t, num_runs=1):
        sig_f, sig_w = self.params['sig_f'], self.params['sig_w']
        f, w, v = z
        dt = (np.roll(t, -1) - t)[:-1]
        f_t_avg = f[None, None, :] + (v - w)[None, None, :]*t[:, None, None]

        f_t = np.random.normal(
            loc=f_t_avg, scale=sig_f/np.sqrt(dt[:, None, None]), size=(len(dt), num_runs, *f.shape))
        w_t = np.random.normal(loc=w[None, None, :], scale=sig_w/np.sqrt(
            dt[:, None, None]), size=(len(dt), num_runs, *f.shape))
        return f_t, w_t

    def get_suff_stats(self, x_t, t_obs):
        f_t, w_t = x_t

        T = f_t.shape[0]
        t_ish = np.arange(1, T+1)/T

        f_avg = np.mean(f_t, axis=0)
        f_avg_ = f_avg[None, :]
        w_avg = np.mean(w_t, axis=0)
        v_rel = (12/t_obs)*np.mean((f_t - f_avg_)*(t_ish - 0.5), axis=0)
        y = f_avg, w_avg, v_rel
        return y

    # has been checked
    def get_det_cov(self, t_obs):
        xi_f, xi_w, xi_v = self.params['xi_f'], self.params['xi_w'], self.params['xi_v']
        sig_f, sig_w = self.params['sig_f'], self.params['sig_w']

        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2

        det = ((Ct + t2_avg/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_v_**2) + 1/(xi_w_**2))
               + (1/(xi_v_**2))*(1 + 1/(xi_f_**2))*((sig_f/sig_w)**2 + 1/xi_w_**2))
        return det

    # has been checked
    def get_det_ratio(self, t_obs):
        xi_f, xi_w = self.params['xi_f'], self.params['xi_w']
        sig_f, sig_w = self.params['sig_f'], self.params['sig_w']

        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2

        det = self.get_det_cov(t_obs)
        det_red = Ct + t2_avg/(xi_f_**2) + (1 + 1/(xi_f_**2)) * \
            ((sig_f/sig_w)**2 + 1/xi_w_**2)

        det_ratio = det/det_red
        return det_ratio

    # has been checked

    def get_decision(self, y, t_obs):
        # decide whether model should be reduced (delta >= 0) or not
        # return decision (1 or 0) and associated confidence (0 <= p <= 1)

        sig_f = self.params['sig_f']
        # required params
        delta_0, xi_v = self.params['delta_0'], self.params['xi_v']
        det_ratio = self.get_det_ratio(t_obs)

        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        extra_factor = t_obs/(sig_f**2)
        mu = self.get_post_v(y, t_obs)

        delta = 0.5*extra_factor*det_ratio * \
            (mu**2) - 0.5*np.log((xi_v_**2)*det_ratio) + delta_0

        delta_abs = np.abs(delta)     # get decision confidence
        p_correct = np.exp(delta_abs)/(1 + np.exp(delta_abs))

        decision = (delta >= 0).astype(int)
        return decision, p_correct

    def get_avg_delta(self, z, t_obs):
        # decide whether model should be reduced (delta >= 0) or not
        # return decision (1 or 0) and associated confidence (0 <= p <= 1)

        sig_f = self.params['sig_f']
        # required params
        delta_0, xi_v = self.params['delta_0'], self.params['xi_v']
        det_ratio = self.get_det_ratio(t_obs)

        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        extra_factor = t_obs/(sig_f**2)
        mu, sig = self.get_avg_mu_sig(z, t_obs)

        delta = 0.5*extra_factor*det_ratio * \
            (mu**2 + sig**2) - 0.5*np.log((xi_v_**2)*det_ratio) + delta_0

        # delta_abs = np.abs(delta)     # get decision confidence
        # p_correct = np.exp(delta_abs)/(1 + np.exp(delta_abs))

        return delta  # , p_correct

    # has been checked
    def get_decision_boundary(self, t_obs):
        # required params
        delta_0, xi_v = self.params['delta_0'], self.params['xi_v']
        sig_f = self.params['sig_f']
        xi_v_ = xi_v*(np.sqrt(t_obs)/sig_f)
        extra_factor = t_obs/(sig_f**2)

        det_ratio = self.get_det_ratio(t_obs)

        thresh_sq = (np.log((xi_v_**2)*det_ratio) - 2*delta_0) / \
            (extra_factor*det_ratio)
        q = np.sqrt(np.maximum(thresh_sq, 0))
        return q

    # has been checked

    def get_avg_mu_sig(self, z, t_obs):
        xi_f, xi_w = self.params['xi_f'], self.params['xi_w']
        sig_f, sig_w = self.params['sig_f'], self.params['sig_w']
        xi_f_ = xi_f*(np.sqrt(t_obs)/sig_f)
        xi_w_ = xi_w*(np.sqrt(t_obs)/sig_f)
        t_avg = t_obs/2
        t2_avg = (t_obs**2)/3
        Ct = t2_avg - t_avg**2
        f0, w, v = z

        prefactor = 1/self.get_det_cov(t_obs)
        mu = prefactor*((t_avg/(xi_f_**2))*((sig_f/sig_w)**2 + 1/xi_w_**2)*f0
                        - (1/xi_w_**2)*(Ct + t2_avg/(xi_f_**2))*w
                        + (Ct + t2_avg/(xi_f_**2))*((sig_f/sig_w)**2 + 1/xi_w_**2)*v)

        sig_inner = ((((t_avg/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_w_**2)))**2)*1.
                     + (((Ct + t2_avg/(xi_f_**2)))**2)*((sig_f/sig_w)**2)
                     + (((1 + 1/(xi_f_**2))*((sig_f/sig_w)**2 + 1/(xi_w_**2)))**2)*(Ct))
        sig = prefactor*(sig_f/np.sqrt(t_obs))*np.sqrt(sig_inner)
        return mu, sig

    # has been checked
    def get_psycho_curve(self, z, t_obs):
        mu, sig = self.get_avg_mu_sig(z, t_obs)
        q = self.get_decision_boundary(t_obs)

        curve = 0.5*(erf((q - mu)/(np.sqrt(2)*sig)) +
                     erf((q + mu)/(np.sqrt(2)*sig)))
        return curve


class BayesDCIDynamic_full():

    def generate_obs(self, z, t, num_runs=1):
        sig_f, sig_w = self.params['sig_f'], self.params['sig_w']
        f, w, v = z
        dt = (np.roll(t, -1) - t)[:-1]
        f_t_avg = f[None, None, :] + (v - w)[None, None, :]*t[:, None, None]

        f_t = np.random.normal(
            loc=f_t_avg, scale=sig_f/np.sqrt(dt[:, None, None]), size=(len(dt), num_runs, *f.shape))
        w_t = np.random.normal(loc=w[None, None, :], scale=sig_w/np.sqrt(
            dt[:, None, None]), size=(len(dt), num_runs, *f.shape))
        return f_t, w_t

    def update(self, x, mu, cov):

        A, M, sig_obs_inv = self.params['A'], self.params['M'], self.params['sig_obs_inv']
        # mu, cov = self.mu, self.cov
        dt = self.params['dt']

        dmu = A @ mu + cov @ M.T @ sig_obs_inv @ (x - M @ mu)
        dcov = A @ cov + cov @ A.T - cov @ M.T @ sig_obs_inv @ M @ cov

        mu += dmu*dt
        cov += dcov*dt
        return mu, cov


# class Bayes
