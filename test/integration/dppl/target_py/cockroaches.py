from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def transformed_data(*, N, exposure2, roach1, senior, treatment, y):
    # Transformed data
    log_expo = zeros([N])
    sqrt_roach = zeros([N])
    log_expo = log(exposure2)
    sqrt_roach = sqrt(roach1)
    return { 'log_expo': log_expo, 'sqrt_roach': sqrt_roach }

def model(*, N, exposure2, roach1, senior, treatment, y, log_expo, sqrt_roach):
    # Parameters
    beta_1 = sample('beta_1', improper_uniform(shape=None))
    beta_2 = sample('beta_2', improper_uniform(shape=None))
    beta_3 = sample('beta_3', improper_uniform(shape=None))
    beta_4 = sample('beta_4', improper_uniform(shape=None))
    # Model
    observe('beta_1__1', normal(0, 5), beta_1)
    observe('beta_2__2', normal(0, 2.5), beta_2)
    observe('beta_3__3', normal(0, 2.5), beta_3)
    observe('beta_4__4', normal(0, 2.5), beta_4)
    observe('y__5', poisson_log(log_expo + beta_1 + beta_2 * sqrt_roach + beta_3 * treatment + beta_4 * senior), y)


