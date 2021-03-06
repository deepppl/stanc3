from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    
    return {  }

def model():
    # Parameters
    cluster = sample('cluster', improper_uniform(shape=[]))
    theta = sample('theta', improper_uniform(shape=[]))
    # Model
    observe('_cluster__1', normal(0, 1), cluster)
    mu = None
    if cluster > 0:
        mu = 2
    else:
        mu = 0
    observe('_theta__2', normal(mu, 1), theta)

