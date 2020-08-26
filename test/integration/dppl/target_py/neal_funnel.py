from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model():
    # Parameters
    y_std = sample('y_std', improper_uniform(shape=None))
    x_std = sample('x_std', improper_uniform(shape=None))
    # Transformed parameters
    y = 3.0 * y_std
    x = exp(y / 2) * x_std
    # Model
    observe('y_std__1', normal(0, 1), y_std)
    observe('x_std__2', normal(0, 1), x_std)

def generated_quantities(*, y_std, x_std):
    # Transformed parameters
    y = 3.0 * y_std
    x = exp(y / 2) * x_std
    return { 'y': y, 'x': x }
