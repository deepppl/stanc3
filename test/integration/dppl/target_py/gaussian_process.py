from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap
from stanpyro.stanlib import exp_real, rep_vector_int_int, square_real

def convert_inputs(inputs):
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_float)
    return { 'N': N, 'x': x }

def transformed_data(*, N, x):
    # Transformed data
    mu = rep_vector_int_int(0, N)
    K = empty([N, N], dtype=dtype_float)
    for i in range(1,(N - 1) + 1):
        K[i - 1, i - 1] = 1 + array(0.1, dtype=dtype_float)
        for j in range((i + 1),N + 1):
            K[i - 1, j - 1] = exp_real(- array(0.5, dtype=dtype_float) * square_real(
                                       x[i - 1] - x[j - 1]))
            K[j - 1, i - 1] = K[i - 1, j - 1]
    K[N - 1, N - 1] = 1 + array(0.1, dtype=dtype_float)
    return { 'mu': mu, 'K': K }

def model(*, N, x, mu, K):
    # Parameters
    y = sample('y', improper_uniform(shape=[N]))
    # Model
    observe('_y__1', multi_normal(mu, K), y)

