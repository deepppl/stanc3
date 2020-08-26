from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model():
    # Parameters
    theta = sample('theta', lower_constrained_improper_uniform(0, shape=None))
    # Model
    observe('expr__1', normal(log(10.0), 1.0), log(theta))
    factor('expr__2', - log(fabs(theta)))
