from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network

def convert_inputs(inputs):
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_long)
    return { 'N': N, 'x': x }

def model(*, N, x):
    # Parameters
    z = sample('z', uniform(0, 1))
    # Model
    observe('z__1', beta(1, 1), z)
    observe('x__2', bernoulli(z), x)

