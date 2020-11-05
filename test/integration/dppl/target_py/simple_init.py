from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float

def convert_inputs(inputs):
    
    return {  }

def model():
    # Parameters
    # Model
    if array(1.0, dtype=dtype_float) == array(0.0, dtype=dtype_float):
        a = array(1.0, dtype=dtype_float)

