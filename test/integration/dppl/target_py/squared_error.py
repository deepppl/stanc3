from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N, K, y, x):
    # Parameters
    beta = sample('beta', improper_uniform(shape=[K]))
    # Transformed parameters
    squared_error = None
    squared_error = dot_self(y - dot(x, beta))
    # Model
    factor('expr__1', - squared_error)


def generated_quantities(*, N, K, y, x, beta):
    # Transformed parameters
    squared_error = None
    squared_error = dot_self(y - dot(x, beta))
    # Generated quantities
    sigma_squared = None
    sigma_squared = squared_error / N
    return { 'squared_error': squared_error, 'sigma_squared': sigma_squared }
