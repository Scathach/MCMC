import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import scipy.constants as con

import emcee
import corner
import scipy.optimize as op
import random



x = np.asarray([3.368,4.618,12.082,22.194,3.6,4.5])   # lam in the mcmc.py file
y = np.asarray([-18.435,-17.042,-15.728,-16.307,-18.063,-17.173]) # logf in the mcmc.py file
yerr = np.asarray([0.087,0.087,0.087,0.087,0.043,0.043]) # errlogf in the mcmc.py file

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

Teffmin = 10.0 #effective temperature minimum
Teffmax = 1000.0 #effective temperature maximum
logfacmin = -100.0 #log factor minimum
logfacmax = 0.0 #log factor maximum
#theatshape is 2 X 2 array
thetashape=np.array([[Teffmin,Teffmax],[logfacmin,logfacmax]])

def model(x, T,logfactor):

    #takes in the wavelength array approximate Temp and the log factor and returns and array of logflux
    wav = x * 1.0e-6
    flux = np.empty([len(wav)])
    logflux = np.empty([len(wav)])

    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    for i in range(len(wav)):
        a = 2.0*h*c**2
        b = h*c/(wav[i]*k*T)
        flux[i] = a/ ( (wav[i]**5) * (np.exp(b) - 1.0) )
        logflux[i] = logfactor + np.log10(flux[i])
    return logflux

def log_like(x,logf,errlogf,theta):
    print(theta)
    print(logf)
    print(errlogf)
    print(x)
    residuals = logf - model(x,theta[0],theta[1])
    loglike=0.0
    for i in range(len(x)):
        loglike = loglike - np.log(errlogf[i]) - 0.5*(residuals[i]/errlogf[i])**2
    loglike = loglike - 0.5*len(x)*np.log(2.0*np.pi)
    return loglike

def log_prior(theta,thetashape):

    logpriors=np.empty([len(theta)])
    #logprior=0.0

    # Prior for theta[0]: Teff~logU[Teffmin,Teffmax]
    Teff = theta[0]
    Teffmin = thetashape[0][0]
    Teffmax = thetashape[0][1]
    if Teffmin < Teff < Teffmax:
        logpriors[0] = ( 1.0/(np.log(Teffmax) - np.log(Teffmin)) )/Teff
    else:
        logpriors[0] = -1.0e99 # -infinity

    # Prior for theta[1]: logfac~U[logfacmin,logfacmax]
    logfac = theta[1]
    logfacmin = thetashape[1][0]
    logfacmax = thetashape[1][1]
    if logfacmin < logfac < logfacmax:
        logpriors[1] = 1.0/(logfacmax - logfacmin)
    else:
        logpriors[1] = -1.0e99 # -infinity

    #logprior = np.sum(logpriors)

    return np.sum(logpriors)

# Initialize the MCMC from a random point drawn from the prior
Teffinitial = np.exp( np.random.uniform(np.log(thetashape[0][0]),np.log(thetashape[0][1])) )
logfacinitial=np.random.uniform(thetashape[1][0],thetashape[1][1])
thetachain=np.array([[Teffinitial,logfacinitial]])

# Calculate the associated modified loglike
loglikechain=np.empty([1])
loglikechain[0]=log_prior(thetachain[0],thetashape) + log_like(x,y,yerr,thetachain[0])


def lnprob(theta, x, y, yerr):
    lp = log_prior(theta,thetashape)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(theta, x, y, yerr)




ndim, nwalkers = 2, 100
pos = [[Teffinitial,logfacinitial] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

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
