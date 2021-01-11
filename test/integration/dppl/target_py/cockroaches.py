from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap
from runtimes.pyro.stanlib import log_vector, sqrt_vector

def convert_inputs(inputs):
    N = inputs['N']
    exposure2 = array(inputs['exposure2'], dtype=dtype_float)
    roach1 = array(inputs['roach1'], dtype=dtype_float)
    senior = array(inputs['senior'], dtype=dtype_float)
    treatment = array(inputs['treatment'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_long)
    return { 'N': N, 'exposure2': exposure2, 'roach1': roach1,
             'senior': senior, 'treatment': treatment, 'y': y }

def transformed_data(*, N, exposure2, roach1, senior, treatment, y):
    # Transformed data
    log_expo = log_vector(exposure2)
    sqrt_roach = sqrt_vector(roach1)
    return { 'log_expo': log_expo, 'sqrt_roach': sqrt_roach }

def model(*, N, exposure2, roach1, senior, treatment, y, log_expo, sqrt_roach):
    # Parameters
    beta_1 = sample('beta_1', improper_uniform(shape=[]))
    beta_2 = sample('beta_2', improper_uniform(shape=[]))
    beta_3 = sample('beta_3', improper_uniform(shape=[]))
    beta_4 = sample('beta_4', improper_uniform(shape=[]))
    # Model
    observe('_beta_1__1', normal(0, 5), beta_1)
    observe('_beta_2__2', normal(0, array(2.5, dtype=dtype_float)), beta_2)
    observe('_beta_3__3', normal(0, array(2.5, dtype=dtype_float)), beta_3)
    observe('_beta_4__4', normal(0, array(2.5, dtype=dtype_float)), beta_4)
    observe('_y__5', poisson_log(log_expo + beta_1 + beta_2 * sqrt_roach + beta_3 * treatment + beta_4 * senior), y)


