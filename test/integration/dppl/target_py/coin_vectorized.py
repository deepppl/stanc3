from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    x = array(inputs['x'], dtype=dtype_long)
    return { 'x': x }

def model(*, x):
    # Parameters
    theta = sample('theta', uniform(0, 1))
    # Model
    observe('_theta__1', uniform(array(0.0, dtype=dtype_float),
                                 array(1.0, dtype=dtype_float)), theta)
    observe('_x__2', bernoulli(theta), x)

