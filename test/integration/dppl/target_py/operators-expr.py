from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network

def convert_inputs(inputs):
    x = array(inputs['x'], dtype=dtype_long)
    return { 'x': x }

def model(*, x):
    # Parameters
    theta = sample('theta', uniform(0, 1))
    # Model
    observe('theta__1', uniform(true_divide(0 * 3, 5), 1 + 5 - 5), theta)
    for i in range(1,10 + 1):
        if (1 <= 10) and (1 > 5 or 2 < 1):
            observe(f'x__{i}__2', bernoulli(theta), x[i - 1])
    print(x)

