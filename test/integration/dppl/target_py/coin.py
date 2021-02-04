from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_long)
    return { 'N': N, 'x': x }

def model(*, N, x):
    # Parameters
    z = sample('z', uniform(0, 1))
    # Model
    observe('_z__1', beta(1, 1), z)
    for i in range(1,N + 1):
        observe(f'_x__{i}__2', bernoulli(z), x[i - 1])

