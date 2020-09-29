from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network

def convert_inputs(inputs):
    N = inputs['N']
    K = inputs['K']
    y = array(inputs['y'], dtype=dtype_float)
    x = array(inputs['x'], dtype=dtype_float)
    return { 'N': N, 'K': K, 'y': y, 'x': x }

def model(*, N, K, y, x):
    # Parameters
    beta = sample('beta', improper_uniform(shape=[K]))
    # Transformed parameters
    squared_error = None
    squared_error = dot_self_vector(y - matmul(x, beta))
    # Model
    factor('expr__1', - squared_error)


def generated_quantities(*, N, K, y, x, beta):
    # Transformed parameters
    squared_error = None
    squared_error = dot_self_vector(y - matmul(x, beta))
    # Generated quantities
    sigma_squared = None
    sigma_squared = true_divide(squared_error, N)
    return { 'squared_error': squared_error, 'sigma_squared': sigma_squared }
