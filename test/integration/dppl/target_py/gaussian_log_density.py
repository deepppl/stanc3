from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    
    return {  }

def model():
    # Parameters
    theta = sample('theta', improper_uniform(shape=[]))
    # Model
    factor('_expr__1', - array(0.5, dtype=dtype_float) * (theta - array(1000.0, dtype=dtype_float)) * (theta - array(1000.0, dtype=dtype_float)))


