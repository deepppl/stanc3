from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    x = array(inputs['x'], dtype=dtype_long)
    return { 'x': x }

def model(*, x):
    # Parameters
    theta = sample('theta', uniform(0, 1))
    # Model
    observe('_x__1', bernoulli(theta), x)
    observe('_theta__2', uniform(0, 1), theta)

