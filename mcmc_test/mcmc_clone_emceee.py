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

cov = np.cov(x,y)

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
samples=np.array([[Teffinitial,logfacinitial]])

# Calculate the associated modified loglike
loglikechain=np.empty([1])
loglikechain[0]=log_prior(samples[0],thetashape) + log_like(x,y,yerr,samples[0])

def lnprob(theta, x, y, yerr):
    lp = log_prior(theta,thetashape)

    loglikechain=np.empty([1])
    loglikechain[0]=log_prior(samples[0],thetashape) + log_like(x,y,yerr,samples[0])
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_like(x,y,yerr,theta) #loglikechain[0]

ndim, nwalkers = 2, 100
pos = [[Teffinitial,logfacinitial] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.MHSampler(cov, dim = ndim, lnprobfn = lnprob, args=(x, y, yerr))


# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos[0], 10000, rstate0=np.random.get_state())
print("Done.")

# Make the triangle plot.
burnin = 500
samples = sampler.chain[burnin:,:].reshape((-1, 2))

# Compute the quantiles.
samples[:] #= np.exp(samples[:])
T_mcmc, logfac_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0))))

print("""MCMC result:
    T = {0[0]} +{0[1]} -{0[2]}
    Log Factor = {1[0]} +{1[1]} -{1[2]}
""".format(T_mcmc, logfac_mcmc))

# ------------------------------------------------------------------------------
# Ploting MCMC
plotting_wavelength = np.arange(x[0], 25.0, 0.01)

# Plot the initial guess results.
plt.errorbar(x, y, yerr=yerr, fmt=".r")
plt.plot(plotting_wavelength, model(plotting_wavelength,samples[0][0],samples[0][1]), "--k", lw=1)
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/Initial Guess Result.png")
plt.close()

# Plot the MCMC results.
plt.errorbar(x, y, yerr=yerr,fmt='.r',ms="6")
plt.plot(plotting_wavelength, model(plotting_wavelength,T_mcmc[0],logfac_mcmc[0]), "--k", lw=1)
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/MCMC results.png")
plt.close()

# Plot
jlist=np.arange(len(samples))
plt.scatter(samples[:,0], samples[:,1], c=jlist, cmap='coolwarm')
plt.xlabel('Temperature [K]')
plt.ylabel('log10(factor)')
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/1B Temp vs logfactor")
plt.close()

np.max(samples[:,1])


plt.plot(samples[:,1])
plt.xlabel('Chain number')
plt.ylabel('loglike')
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/2B Chain number vs loglike")
#plt.show()
plt.close()

loglikeburn=np.median(samples[:,1])
j=-1
while True:
    j=j+1
    if samples[:,1][j] > loglikeburn:
        break
burnj=j
print( 'Burn point = ',burnj)

jlist=np.arange(len(samples))
plt.scatter(samples[burnj:,0], samples[burnj:,1], c=jlist[burnj:], cmap='coolwarm',alpha=0.5)
plt.xlabel('Temperature [K]')
plt.ylabel('log10(factor)')
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/3B Temperatur vs log10(factor) B")
#plt.show()
plt.close()

print( 'Temperature [K] = ',np.round(np.median(samples[burnj:,0]),1),'-',np.round(np.median(samples[burnj:,0])-np.percentile(samples[burnj:,0],15.9),1),'+',np.round(np.percentile(samples[burnj:,0],84.1)-np.median(samples[burnj:,0]),1))
print( 'log10(factor) = ',np.round(np.median(samples[burnj:,1]),3),'-',np.round(np.median(samples[burnj:,1])-np.percentile(samples[burnj:,1],15.9),3),'+',np.round(np.percentile(samples[burnj:,1],84.1)-np.median(samples[burnj:,1]),3))

ascii.write(samples[burnj:,:], "chains.dat")


plt.plot(samples[burnj:,0])
plt.title('Check mixing')
plt.xlabel('Chain number')
plt.ylabel('Temperature [K]')
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/4B Check mixing, Temperature A")
#plt.show()
plt.close()


plt.plot(samples[burnj:,1])
plt.title('Check mixing')
plt.xlabel('Chain number')
plt.ylabel('log10(factor)')
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/5B Check mixing, log10(factor) A")
#plt.show()
plt.close()


temp=np.empty([len(samples)-burnj])
temp[0]=samples[burnj,0]
for i in range(burnj+1,len(samples)):
    temp[i-burnj]=np.mean(samples[burnj:i,0])
plt.plot(temp)
plt.title('Check mixing')
plt.xlabel('Chain number')
plt.ylabel('Temperature [K]')
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/6B Check mixing, Temperature B")
#plt.show()
plt.close()

temp=np.empty([len(samples)-burnj])
temp[0]=samples[burnj,1]
for i in range(burnj+1,len(samples)):
    temp[i-burnj]=np.mean(samples[burnj:i,1])
plt.plot(temp)
plt.title('Check mixing')
plt.xlabel('Chain number')
plt.ylabel('log10(factor)')
plt.savefig("C:/Users/Misha Savchenko/coding/MCMC_Fork/MCMC/mcmc_test/emcee_model/7B Check mixing, log10(factor) B")
#plt.show()
plt.close()
