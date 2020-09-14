from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model():
    # Parameters
    theta = sample('theta', improper_uniform(shape=None))
    # Model
    observe('theta__1', normal(1000.0, 1.0), theta)
    observe('theta__2', normal(1000.0, 1.0), theta)

