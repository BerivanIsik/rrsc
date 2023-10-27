#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy import stats as st
import scipy.special as sc
from scipy.stats import norm
import math
import random

''' PrivUnitG -- we use the PrivUnitG algorithm in https://arxiv.org/abs/2205.02466.'''
def PrivUnitG_n_users(X, eps):
    (d, n) = X.shape
    x_perturbed = np.zeros((d, n))
    p = None
    gamma = None
    sigma = None
    for i in range(n):
        u = X[:, i]
        x_perturbed[:, i], p, gamma, sigma = PrivUnitG(u, eps, p, gamma, sigma)
    return x_perturbed

def PrivUnitG(x, eps, p=None, gamma=None, sigma=None, n_tries=None):
    if not p:
        p = priv_unit_G_get_p(eps)
    if p is None or gamma is None or sigma is None:
        gamma, sigma = get_gamma_sigma(p, eps)
    dim = x.size
    g = np.random.normal(0, 1, size=dim)
    pos_cor = np.random.binomial(1, p)

    if pos_cor:
        chosen_dps = np.array([sample_from_G_tail_stable(gamma)])
    else:
        if n_tries is None:
            n_tries = 25  # here probability of success is 1/2
        dps = np.random.normal(0, 1, size=n_tries)
        chosen_dps = dps[dps < gamma]

    if chosen_dps.size == 0:
        print('failure')
        return g * sigma
    target_dp = chosen_dps[0]

    yperp = g - (g.dot(x)) * x
    ypar = target_dp * x
    return sigma * (yperp + ypar), p, gamma, sigma


def get_gamma_sigma(p, eps):
    # Want p(1-q)/q(1-p) = exp(eps)
    # I.e q^{-1} -1 = (1-q)/q = exp(eps) * (1-p)/p
    qinv = 1 + (math.exp(eps) * (1.0 - p) / p)
    q = 1.0 / qinv
    gamma = st.norm.isf(q)
    # Now the expected dot product is (1-p)*E[N(0,1)|<gamma] + pE[N(0,1)|>gamma]
    # These conditional expectations are given by pdf(gamma)/cdf(gamma) and pdf(gamma)/sf(gamma)
    unnorm_mu = st.norm.pdf(gamma) * (-(1.0 - p) / st.norm.cdf(gamma) + p / st.norm.sf(gamma))
    sigma = 1. / unnorm_mu
    return gamma, sigma


def priv_unit_G_get_p(eps, return_sigma=False):
    # Mechanism:
    # With probability p, sample a Gaussian conditioned on g.x \geq gamma
    # With probability (1-p), sample conditioned on g.x \leq gamma
    # Scale g appropriately to get the expectation right
    # Let q(gamma) = Pr[g.x \geq gamma] = Pr[N(0,1) \geq gamma] = st.norm.sf(gamma)
    # Then density for x above threshold = p(x)  * p/q(gamma)
    # And density for x below threhsold = p(x) * (1-p)/(1-q(gamma))
    # Thus for a p, gamma is determined by the privacy constraint.
    plist = np.arange(0.01, 1.0, 0.01)
    glist = []
    slist = []
    for p in plist:
        gamma, sigma = get_gamma_sigma(p, eps)
        # thus we have to scale this rv by sigma to get it to be unbiased
        # The variance proxy is then d sigma^2
        slist.append(sigma)
        glist.append(gamma)
    ii = np.argmin(slist)
    if return_sigma:
        return plist[ii], slist[ii]
    else:
        return plist[ii]


# More stable version. Works at least until 1000
def sample_from_G_tail_stable(gamma):
    #return sample_from_G_tail(gamma)
    logq = norm.logsf(gamma)
    u = np.random.uniform(low=0, high=1)
    logu = np.log(u)
    logr = logq + logu  # r is now uniform in (0,q)
    # print(q,r)
    return -sc.ndtri(np.exp(logr))



