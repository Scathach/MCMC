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

def model(x, T):
    a = 2.0*h*c**2
    b = h*c/(x*k*T)
    intensity = a/ ( (x**5) * (np.exp(b) - 1.0) )
    return np.log10(intensity)

def log_like(lam,logf,errlogf,theta):
    residuals = logf - model(lam,theta[0],theta[1])
    loglike=0.0
    for i in range(len(lam)):
        loglike = loglike - np.log(errlogf[i]) - 0.5*(residuals[i]/errlogf[i])**2
    loglike = loglike - 0.5*len(lam)*np.log(2.0*np.pi)
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
