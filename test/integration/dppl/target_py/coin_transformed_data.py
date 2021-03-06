from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    x = array(inputs['x'], dtype=dtype_long)
    return { 'x': x }

def transformed_data(*, x):
    # Transformed data
    y = empty([10], dtype=dtype_long)
    for i in range(1,10 + 1):
        y[i - 1] = 1 - x[i - 1]
    return { 'y': y }

def model(*, x, y):
    # Parameters
    theta = sample('theta', uniform(0, 1))
    # Model
    observe('_theta__1', uniform(0, 1), theta)
    for i in range(1,10 + 1):
        observe(f'_y__{i}__2', bernoulli(theta), y[i - 1])

