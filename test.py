#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

# Reproducible results!
np.random.seed(123)

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.

y_list = []
x_list = []
yerr_list = []

N = 50
x = np.sort(10*np.random.rand(N)) #generates 50 random x values
yerr = 0.1+0.5*np.random.rand(N) #generates 50 random y error values
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)
print(y)
"""
y_list.append(y)
x_list.append(x)
yerr_list.append(yerr)

print(y_list)
print(x_list)
print(yerr_list)
"""
# Plot the dataset and the true model.
xl = np.array([0, 10])
pl.errorbar(x, y, yerr=yerr, fmt=".k")
pl.plot(xl, m_true*xl+b_true, "k", lw=3, alpha=0.6)
pl.ylim(-9, 9)
pl.xlabel("$x$")
pl.ylabel("$y$")
pl.tight_layout()
pl.savefig("line-data.png")
