from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N_obs, N_mis, y_obs):
    # Parameters
    mu = sample('mu', improper_uniform(shape=None))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=None))
    y_mis = sample('y_mis', improper_uniform(shape=[N_mis]))
    # Model
    observe('y_obs__1', normal(mu, sigma), y_obs)
    observe('y_mis__2', normal(mu, sigma), y_mis)

