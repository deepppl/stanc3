from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float
from runtimes.pyro.stanlib import exp_real

def convert_inputs(inputs):
    
    return {  }

def model():
    # Parameters
    y_std = sample('y_std', improper_uniform(shape=[]))
    x_std = sample('x_std', improper_uniform(shape=[]))
    # Transformed parameters
    y = array(3.0, dtype=dtype_float) * y_std
    x = exp_real(true_divide(y, 2)) * x_std
    # Model
    observe('y_std__1', normal(0, 1), y_std)
    observe('x_std__2', normal(0, 1), x_std)


def generated_quantities(*, y_std, x_std):
    # Transformed parameters
    y = array(3.0, dtype=dtype_float) * y_std
    x = exp_real(true_divide(y, 2)) * x_std
    return { 'y': y, 'x': x }

def map_generated_quantities(_samples, ):
    def _generated_quantities(y_std, x_std):
        return generated_quantities(y_std=y_std, x_std=x_std)
    return vmap(_generated_quantities)(_samples['y_std'], _samples['x_std'])
