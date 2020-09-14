from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model():
    # Parameters
    cluster = sample('cluster', improper_uniform(shape=None))
    theta = sample('theta', improper_uniform(shape=None))
    # Model
    mu = None
    observe('cluster__1', normal(0, 1), cluster)
    if cluster > 0:
        mu = 2
    else:
        mu = 0
    observe('theta__2', normal(mu, 1), theta)

