#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import random

x = np.asarray([3.368,4.618,12.082,22.194,3.6,4.5])
y = np.asarray([-18.435,-17.042,-15.728,-16.307,-18.063,-17.173])
yerr = np.asarray([0.087,0.087,0.087,0.087,0.043,0.043])

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

# Initialize the MCMC from a random point drawn from the prior
Teffinitial = np.exp( np.random.uniform(np.log(10),np.log(1000)) )
logfacinitial=np.random.uniform(-100,0)
thetachain=np.array([[Teffinitial,logfacinitial]])
T_ls = thetachain[0]
# Calculate the associated modified loglike
#loglikechain=np.empty([1])
#loglikechain[0]=log_prior(thetachain[0],thetashape) + log_like(lam,logf,errlogf,thetachain[0])

def planck(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return np.log10(intensity)

# Define the probability function as likelihood * prior.
def lnprior(theta):
    logpriors = np.empty([len(theta)])
    T, lnf = theta
    if 10 < T < 1000:
        logpriors[0] = (1.0/(np.log(1000) - np.log(10))/T)
    else:
        logpriors[0] = -np.inf
    if -100 < lnf < 0:
        logpriors[1] = 1/(0-(-100))
    else:
        logpriors[1] = -np.inf

    return np.sum(logpriors)

def lnlike(theta, x, y, yerr):
    T, lnf = theta

    h = 6.626e-34
    c = 3.0e+8
    k = 1.38e-23

    a = 2.0*h*c**2
    b = h*c/(x*k*T)
    model =np.log10( a/ ( (x**5) * (np.exp(b) - 1.0) ))

    inv_sigma2 = 1.0/(yerr**2 + planck(x,T)**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
print(T_ls,"T_ls")

# Find the maximum likelihood value.
chi2 = lambda *args: -2 * lnlike(*args)
result = op.minimize(chi2, T_ls, args=(x, y, yerr), bounds=((0,1000),(-100,1)))
T_ml = result["x"]
print("""#Maximum likelihood result:
    #T = {0}
""".format(T_ml))


# Set up the sampler.
ndim, nwalkers = 2, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, 500, rstate0=np.random.get_state())
print("Done.")

# Make the triangle plot.
burnin = 50
samples = sampler.chain[:,burnin:, :].reshape((-1, 1))
# Compute the quantiles.
samples[:] = np.exp(samples[:])
T_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0))))

print("""MCMC result:
    T = {0[0][0]} +{0[0][1]} -{0[0][2]}
""".format(T_mcmc))
