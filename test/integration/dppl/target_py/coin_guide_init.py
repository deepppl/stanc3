from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N, x):
    # Parameters
    theta = sample('theta', uniform(0.0, 1.0))
    # Model
    observe('theta__1', beta(10.0, 10.0), theta)
    for i in range(1,10 + 1):
        observe(f'x__{i}__2', bernoulli(theta), x[i - 1])

def guide(*, N, x):
    # Guide Parameters
    alpha_q = param('alpha_q', array(15.0))
    beta_q = param('beta_q', array(15.0))
    # Guide
    theta = sample('theta', beta(alpha_q, beta_q))
