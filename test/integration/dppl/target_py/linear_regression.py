from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_float)
    return { 'N': N, 'x': x, 'y': y }

def model(*, N, x, y):
    # Parameters
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_y__1', normal(alpha + beta__ * x, sigma), y)

