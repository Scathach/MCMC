from mcmc_clone import model
import pytest

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import scipy.constants as con

import emcee
import corner
import scipy.optimize as op
import random

def model_format_test():

    assert model()
