from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_long)
    return { 'N': N, 'x': x }

def model(*, N, x):
    # Parameters
    theta = sample('theta', uniform(array(0.0, dtype=dtype_float), array(1.0, dtype=dtype_float)))
    # Model
    observe('_theta__1', beta(array(10.0, dtype=dtype_float),
                              array(10.0, dtype=dtype_float)), theta)
    for i in range(1,10 + 1):
        observe(f'_x__{i}__2', bernoulli(theta), x[i - 1])

def guide(*, N, x):
    # Guide Parameters
    alpha_q = param('alpha_q', array(array(15.0, dtype=dtype_float)))
    beta_q = param('beta_q', array(array(15.0, dtype=dtype_float)))
    # Guide
    theta = sample('theta', beta(alpha_q, beta_q))
    return { 'theta': theta }
