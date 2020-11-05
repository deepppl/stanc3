from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float
from runtimes.pyro.stanlib import log_real

def convert_inputs(inputs):
    
    return {  }

def model():
    # Parameters
    theta = sample('theta', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('expr__1', normal(log_real(array(10.0, dtype=dtype_float)),
                              array(1.0, dtype=dtype_float)), log_real(
    theta))
    factor('expr__2', - log_real(fabs_real(theta)))

