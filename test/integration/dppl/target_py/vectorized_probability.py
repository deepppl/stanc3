from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, K, N, x, y):
    # Parameters
    beta = sample('beta', improper_uniform(shape=[K]))
    # Model
    observe('y__1', normal(dot(x, beta), 1), y)

