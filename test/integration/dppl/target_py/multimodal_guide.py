from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float
from runtimes.pyro.stanlib import exp_real

def convert_inputs(inputs):
    
    return {  }

def model():
    # Parameters
    cluster = sample('cluster', improper_uniform(shape=[]))
    theta = sample('theta', improper_uniform(shape=[]))
    # Model
    observe('cluster__1', normal(0, 1), cluster)
    mu = None
    if cluster > 0:
        mu = 2
    else:
        mu = 0
    observe('theta__2', normal(mu, 1), theta)

def guide():
    # Guide Parameters
    mu_cluster = param('mu_cluster', improper_uniform(shape=[]).sample())
    mu1 = param('mu1', improper_uniform(shape=[]).sample())
    mu2 = param('mu2', improper_uniform(shape=[]).sample())
    log_sigma1 = param('log_sigma1', improper_uniform(shape=[]).sample())
    log_sigma2 = param('log_sigma2', improper_uniform(shape=[]).sample())
    # Guide
    cluster = sample('cluster', normal(mu_cluster, 1))
    if cluster > 0:
        theta = sample('theta', normal(mu1, exp_real(log_sigma1)))
    else:
        theta = sample('theta', normal(mu2, exp_real(log_sigma2)))
    return { 'cluster': cluster, 'theta': theta }
