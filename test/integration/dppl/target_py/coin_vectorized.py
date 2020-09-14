from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, x):
    # Parameters
    theta = sample('theta', uniform(0, 1))
    # Model
    observe('theta__1', uniform(0.0, 1.0), theta)
    observe('x__2', bernoulli(theta), x)

