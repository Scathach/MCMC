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

np.seterr(divide='ignore', invalid='ignore')
np.seterr(over='ignore', invalid='ignore')

x = np.asarray([3.368,4.618,12.082,22.194,3.6,4.5])
y = np.asarray([-18.435,-17.042,-15.728,-16.307,-18.063,-17.173])
yerr = np.asarray([0.087,0.087,0.087,0.087,0.043,0.043])

h = 6.626070040*(10**(-34))  #J*s
c = 299792458 #m/s
k_b = 1.38064852*(10**(-23))  #m^2*kg*s^-2*K^-1

a = 2*h*(c**2)
b = (h*c)/k_b

xl = np.array([1, 25])
T_ls = random.randrange(0,1500)
print(T_ls)

logPlanck = np.log10((a/xl**5)*(1/((np.exp(b/(T_ls/xl))-1))))

# Define the probability function as likelihood * prior.

def lnprior(theta):
    T, lnf = theta
    if 0.0 < T < 2000.0 and 0 < lnf < 5:
        return 0.0
    return -np.inf

def lnlike(theta, x, y, yerr):
    T, lnf = theta
    #print(theta,x,y,yerr,"here it is")
    model = np.log10((a/x**5)*(1/((np.exp(b/(x*T))-1))))
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

# Find the maximum likelihood value.
chi2 = lambda *args: -2 * lnlike(*args)
result = op.minimize(chi2, [T_ls,0], args=(x, y, yerr))
print(result)
T_ml = result["x"]
print("""#Maximum likelihood result:
    #T = {0}
""".format(T_ml))

# Plot the maximum likelihood result.
pl.plot(xl, np.log10((a/xl**5)*(1/((np.exp(b/(xl*T_ml))-1)))), "k", lw=2)
pl.savefig("line-max-likelihood.png")


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
print(samples)
# Compute the quantiles.
samples[:] = np.exp(samples[:])
T_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0))))

print("""MCMC result:
    T = {0[0][0]} +{0[0][1]} -{0[0][2]}
""".format(T_mcmc))
