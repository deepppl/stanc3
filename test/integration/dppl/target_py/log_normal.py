from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap
from stanpyro.stanlib import log_real

def convert_inputs(inputs):
    
    return {  }

def model():
    # Parameters
    theta = sample('theta', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_expr__1', normal(log_real(array(10.0, dtype=dtype_float)),
                               array(1.0, dtype=dtype_float)), log_real(
    theta))
    factor('_expr__2', - log_real(fabs_real(theta)))

