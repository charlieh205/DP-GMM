#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:10:24 2022

@author: jeffreymayolo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from ldp_gmm import LDPGMM

df = pd.read_csv("../data/old_faithful.csv")
faith_gmm = LDPGMM(n_components = 2)
faith_gmm.fit(np.array(df['waiting']).reshape(-1,1))
faith_gmm.plot_mixture(bins = 20)
true_params = faith_gmm.get_params()

true_dist1 = (faith_gmm.means_[0][0],faith_gmm.covariances_[0][0][0], faith_gmm.weights_[0])
true_dist2 = (faith_gmm.means_[1][0], faith_gmm.covariances_[1][0][0], faith_gmm.weights_[1])

print(f' True Means: {faith_gmm.means_[0][0], faith_gmm.means_[1][0]}')
print(f' True Covariances: {faith_gmm.covariances_[0][0], faith_gmm.covariances_[1][0]}')
print(f' True weights: {faith_gmm.weights_[0], faith_gmm.weights_[1]}')

ldp_model = LDPGMM(n_components = 2, epsilon = 4)
ldp_model.fit_ldp(np.array(df['waiting']).reshape(-1,1))
ldp_model.plot_mixture(bins = 20)
dp_params = ldp_model.get_params()

dp_dist1 = (ldp_model.means_[0][0],ldp_model.covariances_[0][0][0], ldp_model.weights_[0])
dp_dist2 = (ldp_model.means_[1][0],ldp_model.covariances_[1][0][0], ldp_model.weights_[1])

print(f' TLDP Means: {ldp_model.means_[0][0], ldp_model.means_[1][0]}')
print(f' TLDP MCovariances: {ldp_model.covariances_[0][0], ldp_model.covariances_[1][0]}')
print(f' TLDP Means: {ldp_model.weights_[0], ldp_model.weights_[1]}')


x = np.linspace(np.array(df['waiting']).min(), np.array(df['waiting']).max(), 100)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(df['waiting'], bins=60, density=True, color='gray', alpha=0.5, label='True wait times')
ax.plot(x, true_dist1[2] * sp.stats.norm(true_dist1[0], true_dist1[1]**0.5).pdf(x), color='blue', label='True First Gaussian')
ax.plot(x, true_dist2[2] * sp.stats.norm(true_dist2[0], true_dist2[1]**0.5).pdf(x), color='darkblue', label='True Second Gaussian')
ax.plot(x, dp_dist1[2] * sp.stats.norm(dp_dist1[0], dp_dist1[1]**0.5).pdf(x), color='red', label='DP First Gaussian', linestyle = '--')
ax.plot(x, dp_dist2[2] * sp.stats.norm(dp_dist2[0], dp_dist2[1]**0.5).pdf(x), color='darkred', label='DP Second Gaussian', linestyle = '--')
ax.set_title('GMM for DP geyser wait times')
ax.legend(loc='best')
plt.show()

# Simulations
def get_params(data, n_components, epsilon, DP = False):
    if DP:
        model = LDPGMM(n_components = n_components, epsilon = epsilon)
        model.fit_ldp(data.reshape(-1,1))
        
        dists = np.zeros((n_components, 3))
        for i in range(n_components):
            dists[i,:] = [model.means_[i][0],model.covariances_[i][0][0], model.weights_[i]]
    else:
        model = LDPGMM(n_components = n_components)
        model.fit(data.reshape(-1,1))
        
        dists = np.zeros((n_components, 3))
        for i in range(n_components):
            dists[i,:] = [model.means_[i][0],model.covariances_[i][0][0], model.weights_[i]]
    return dists


# Simulations
samples = 20
components = 3
epsilon = 4
iters = 10

ns = np.linspace(500,10000, samples).astype(int)

dp_dist1 = np.zeros((samples,components))
dp_dist2 = np.zeros((samples,components))
dp_dist3 = np.zeros((samples,components))

true_dist1 = np.zeros((samples,components))
true_dist2 = np.zeros((samples,components))
true_dist3 = np.zeros((samples,components))

dp_dists_list = [dp_dist1, dp_dist2, dp_dist3]
true_dists_list = [true_dist1, true_dist2, true_dist3]

for i,n in enumerate(ns):
    #Create random data with distributions
    pis = [0.2, 0.4, 0.4]
    mus = [17, 30, 50]
    sigmas = [5**2, 4**2, 5**2]
    K = len(pis)
    zs = np.random.choice(np.arange(K), size=n, p=pis)
    # Make hist integers because that's what the local model handles
    y = np.array([np.random.normal(mus[z], sigmas[z]**0.5, 1)[0] for z in zs]).astype(int)
    
    true_dists = get_params(y, components, epsilon)
    
    dist1_trial = np.zeros((iters,components))
    dist2_trial = np.zeros((iters,components))
    dist3_trial = np.zeros((iters,components))
    #dp_dists = get_params(y, components, epsilon, DP = True)
    for k in range(iters):
        dp_dists = get_params(y, components, epsilon, DP = True)
        dist1_trial[k,:] = dp_dists[0,:]
        dist2_trial[k,:] = dp_dists[1,:]
        dist3_trial[k,:] = dp_dists[2,:]
    for j in range(components):
        true_dists_list[j][i,:] = true_dists[j,:]
    dp_dist1[i,:] = np.mean(dist1_trial, axis = 0)
    dp_dist2[i,:] = np.mean(dist2_trial, axis = 0)
    dp_dist3[i,:] = np.mean(dist3_trial, axis = 0)
        
    
    print(str(i) + ' finished')
    
    
mu1_bias = abs(dp_dist1[:,0] - true_dist1[:,0])
mu2_bias = abs(dp_dist2[:,0] - true_dist2[:,0])
mu3_bias = abs(dp_dist3[:,0] - true_dist3[:,0])

pi1_bias = abs(dp_dist1[:,2] - true_dist1[:,2])
pi2_bias = abs(dp_dist2[:,2] - true_dist2[:,2])
pi3_bias = abs(dp_dist3[:,2] - true_dist3[:,2])

sigma1_bias = abs(dp_dist1[:,1] - true_dist1[:,1])
sigma2_bias = abs(dp_dist2[:,1] - true_dist2[:,1])
sigma3_bias = abs(dp_dist3[:,1] - true_dist3[:,1])
    
    
fig, ax = plt.subplots(1,3,figsize = (30,10))

ax[1].plot(ns, mu1_bias, label = '$\mu_1$ Bias')
ax[1].plot(ns, mu2_bias, label = '$\mu_2$ Bias')
ax[1].plot(ns, mu3_bias, label = '$\mu_3$ Bias')
ax[1].set_ylabel('bias', fontsize = 12)
ax[1].set_xlabel('sample size', fontsize = 12)
ax[1].set_title('Mean Bias', fontsize = 15)
ax[1].legend(loc = 'best')

ax[0].plot(ns, pi1_bias, label = '$\pi_1$ Bias')
ax[0].plot(ns, pi2_bias, label = '$\pi_2$ Bias')
ax[0].plot(ns, pi3_bias, label = '$\pi_3$ Bias')
ax[0].set_ylabel('bias', fontsize = 12)
ax[0].set_xlabel('sample size', fontsize = 12)
ax[0].set_title('Proportion Bias')
ax[0].legend(loc = 'best', fontsize = 15)

ax[2].plot(ns, sigma1_bias, label = '$\sigma_1$ Bias')
ax[2].plot(ns, sigma2_bias, label = '$\sigma_2$ Bias')
ax[2].plot(ns, sigma3_bias, label = '$\sigma_3$ Bias')
ax[2].set_ylabel('bias', fontsize = 12)
ax[2].set_xlabel('sample size', fontsize = 12)
ax[2].set_title('Standard Error Bias')
ax[2].legend(loc = 'best', fontsize = 15)

plt.show()
    