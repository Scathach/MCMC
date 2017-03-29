import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import scipy.constants as con

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
