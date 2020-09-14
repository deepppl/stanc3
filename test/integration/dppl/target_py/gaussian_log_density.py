from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model():
    # Parameters
    theta = sample('theta', improper_uniform(shape=None))
    # Model
    factor('expr__1', - 0.5 * (theta - 1000.0) * (theta - 1000.0))

