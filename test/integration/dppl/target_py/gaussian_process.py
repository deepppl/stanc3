from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network
from runtimes.pyro.stanlib import exp_real, rep_vector_int_int, square_real

def convert_inputs(inputs):
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_float)
    return { 'N': N, 'x': x }

def transformed_data(*, N, x):
    # Transformed data
    K = zeros([N, N])
    mu = rep_vector_int_int(0, N)
    for i in range(1,(N - 1) + 1):
        K[i - 1, i - 1] = 1 + array(0.1, dtype=dtype_float)
        for j in range((i + 1),N + 1):
            K[i - 1, j - 1] = exp_real(- array(0.5, dtype=dtype_float) * square_real(
                                       x[i - 1] - x[j - 1]))
            K[j - 1, i - 1] = K[i - 1, j - 1]
    K[N - 1, N - 1] = 1 + array(0.1, dtype=dtype_float)
    return { 'K': K, 'mu': mu }

def model(*, N, x, K, mu):
    # Parameters
    y = sample('y', improper_uniform(shape=[N]))
    # Model
    observe('y__1', multi_normal(mu, K), y)

