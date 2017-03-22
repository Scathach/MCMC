from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import PyAstronomy
import scipy.constants as con
h = 6.626070040*(10**(-34))  #J*s
c = 299792458 #m/s
k_b = 1.38064852*(10**(-23))  #m^2*kg*s^-2*K^-1

a = 2*h*(c**2)
b = (h*c)/k_b

"""x = np.sort(np.asarray([3.368,4.618,12.082,22.194,3.6,4.5]))
y = np.asarray([-18.435,-17.042,-15.728,-16.307,-18.063,-17.173])
yerr = np.asarray([0.087,0.087,0.087,0.087,0.043,0.043])

"""
def logPlanck(x,T_ls):
    return(np.log10((a/x**5)*(1/((np.exp(b/(T_ls/x))-1)))))

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

def planck(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

def model(wavelength,Teff):
    flux = ( (2.0*con.h*con.c**2)/(wavelength**5) )/( np.exp( (con.h*con.c)/(con.k*Teff*wavelength) ) - 1.0 )
    return np.log10(flux)

"""
popt, pcov = curve_fit(,x,y)
print(popt,"popt",pcov,"pcov",logPlanck(x, *pcov),logPlanck(x, *popt))
pl.plot(x,logPlanck(x, *pcov), "r-", label='fit')
pl.plot(x,logPlanck(x, *popt), "r-", label='fit2')
pl.show()"""


x = np.asarray([3.368,4.618,12.082,22.194,3.6,4.5])
y = np.asarray([-18.435,-17.042,-15.728,-16.307,-18.063,-17.173])
yerr = np.asarray([0.087,0.087,0.087,0.087,0.043,0.043])

pl.errorbar(x, y, yerr=yerr, fmt=".k")

X = np.linspace(0,30,256,endpoint=True)
Y1 = model(X,1500)
#Y2 = logPlanck(X,3000)
pl.plot(X,Y1)
#pl.plot(X,Y2)

pl.show()
