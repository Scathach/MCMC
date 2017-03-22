#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from astropy.io import ascii
import scipy.constants as con

data = [[3.368,-18.435,0.087],[4.618,-17.042,0.087],[12.082,-15.728,0.087],[22.194,-16.307,0.087
],[3.6,-18.063,0.043],[4.5,-17.173,0.043]]

x = np.asarray([3.368,4.618,12.082,22.194,3.6,4.5])
y = np.asarray([-18.435,-17.042,-15.728,-16.307,-18.063,-17.173])
yerr = np.asarray([0.087,0.087,0.087,0.087,0.043,0.043])



def model(microns, Teff,logfactor):
    wavelength = microns*1.0e-6
    flux=np.empty([len(wavelength)])
    logflux=np.empty([len(wavelength)])
    for i in range(len(wavelength)):
      # Flux is just the Planck function
      flux = ( (2.0*con.h*con.c**2)/(wavelength[i]**5) )/( np.exp( (con.h*con.c)/(con.k*Teff*wavelength[i]) ) - 1.0 )
      # So logflux (which is what we want) is just the log of this
      logflux = logfactor + np.log10(flux)
      print(flux,np.log10(flux))

    return (logflux)

model(x,y,yerr)
