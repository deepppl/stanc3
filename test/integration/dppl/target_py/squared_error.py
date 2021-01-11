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


def generated_quantities(*, N, y, K, x, beta__):
    # Transformed parameters
    squared_error = dot_self_vector(y - matmul(x, beta__))
    # Generated quantities
    sigma_squared = true_divide(squared_error, N)
    return { 'squared_error': squared_error, 'sigma_squared': sigma_squared }

def map_generated_quantities(_samples, *, N, y, K, x):
    def _generated_quantities(beta__):
        return generated_quantities(N=N, y=y, K=K, x=x, beta__=beta__)
    return vmap(_generated_quantities)(_samples['beta'])
