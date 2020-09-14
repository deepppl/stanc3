from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N, x):
    # Parameters
    z = sample('z', uniform(0, 1))
    # Model
    observe('z__1', beta(1, 1), z)
    observe('x__2', bernoulli(z), x)

def guide(*, N, x):
    # Guide Parameters
    alpha_q = param('alpha_q', lower_constrained_improper_uniform(0, shape=None).sample())
    beta_q = param('beta_q', lower_constrained_improper_uniform(0, shape=None).sample())
    # Guide
    z = sample('z', beta(alpha_q, beta_q))
