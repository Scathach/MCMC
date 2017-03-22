from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import PyAstronomy

np.seterr(divide='ignore', invalid='ignore')
np.seterr(over='ignore', invalid='ignore')

h = 6.626070040*(10**(-34))  #J*s
c = 299792458 #m/s
k_b = 1.38064852*(10**(-23))  #m^2*kg*s^-2*K^-1

a = 2*h*(c**2)
b = (h*c)/k_b

def my_own_Planck(x,T):
    #x is the wavelength
    #returns spectral radiance per sr
    return((a/x**5)*(1/((np.exp(b/(T*x))-1))))

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

def planck(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

X = np.linspace(0,3,256,endpoint=True)

Y1_A = (planck(X*10**-6,3000))
Y2_A = (planck(X*10**-6,4000))
Y3_A = (planck(X*10**-6,5000))

T1 = 750
T2 = 1000
T3 = 1200

plot1, = pl.plot(X,Y1_A, color="red", label=str(T1)+"K")
plot2, = pl.plot(X,Y2_A, color="green", label=str(T2)+"K")
plot3, = pl.plot(X,Y3_A, color="blue", label=str(T3)+"K")
pl.legend([plot1,plot2,plot3],[str(T1)+"K",str(T2)+"K",str(T3)+"K"])
pl.xlabel("Wavelength (um)")
pl.ylabel("Spectral Radiance")

pl.show()
