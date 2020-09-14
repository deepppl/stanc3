from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N, x, y):
    # Parameters
    alpha = sample('alpha', improper_uniform(shape=None))
    beta = sample('beta', improper_uniform(shape=None))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=None))
    # Model
    observe('y__1', normal(alpha + beta * x, sigma), y)

