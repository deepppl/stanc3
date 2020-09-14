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

def guide():
    # Guide Parameters
    mu_cluster = param('mu_cluster', improper_uniform(shape=None).sample())
    mu1 = param('mu1', improper_uniform(shape=None).sample())
    mu2 = param('mu2', improper_uniform(shape=None).sample())
    log_sigma1 = param('log_sigma1', improper_uniform(shape=None).sample())
    log_sigma2 = param('log_sigma2', improper_uniform(shape=None).sample())
    # Guide
    cluster = sample('cluster', normal(mu_cluster, 1))
    if cluster > 0:
        theta = sample('theta', normal(mu1, exp(log_sigma1)))
    else:
        theta = sample('theta', normal(mu2, exp(log_sigma2)))
