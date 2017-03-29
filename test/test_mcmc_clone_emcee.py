from mcmc_test.mcmc_clone import model
import pytest

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import scipy.constants as con

import emcee
import corner
import scipy.optimize as op
import random

from inspect import signature

def test_model_format():
    sig = signature(model)
    params = sig.parameters
    print(len(params))
    assert len(params) == 3
