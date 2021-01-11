from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    K = inputs['K']
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_float)
    return { 'K': K, 'N': N, 'x': x, 'y': y }

def model(*, K, N, x, y):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[K]))
    # Model
    observe('_y__1', normal(matmul(x, beta__), 1), y)

