from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float
from runtimes.pyro.stanlib import dot_self_vector

def convert_inputs(inputs):
    N = inputs['N']
    y = array(inputs['y'], dtype=dtype_float)
    K = inputs['K']
    x = array(inputs['x'], dtype=dtype_float)
    return { 'N': N, 'y': y, 'K': K, 'x': x }

def model(*, N, y, K, x):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[K]))
    # Transformed parameters
    squared_error = dot_self_vector(y - matmul(x, beta__))
    # Model
    factor('expr__1', - squared_error)


def generated_quantities(__inputs__):
    N = __inputs__['N']
    y = __inputs__['y']
    K = __inputs__['K']
    x = __inputs__['x']
    beta__ = __inputs__['beta']
    # Transformed parameters
    squared_error = dot_self_vector(y - matmul(x, beta__))
    # Generated quantities
    sigma_squared = true_divide(squared_error, N)
    return { 'squared_error': squared_error, 'sigma_squared': sigma_squared }
