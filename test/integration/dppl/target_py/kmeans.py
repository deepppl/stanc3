from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float
from runtimes.pyro.stanlib import dot_self_vector, log_int, log_sum_exp_array

def convert_inputs(inputs):
    K = inputs['K']
    N = inputs['N']
    D = inputs['D']
    y = array(inputs['y'], dtype=dtype_float)
    return { 'K': K, 'N': N, 'D': D, 'y': y }

def transformed_data(*, K, N, D, y):
    # Transformed data
    neg_log_K = - log_int(K)
    return { 'neg_log_K': neg_log_K }

def model(*, K, N, D, y, neg_log_K):
    # Parameters
    mu = sample('mu', improper_uniform(shape=[K, D]))
    # Transformed parameters
    soft_z = empty([N, K], dtype=dtype_float)
    for n in range(1,N + 1):
        for k in range(1,K + 1):
            soft_z[n - 1, k - 1] = neg_log_K - array(0.5, dtype=dtype_float) * dot_self_vector(
            mu[k - 1] - y[n - 1])
    # Model
    for k in range(1,K + 1):
        observe(f'mu__{k}__1', normal(0, 1), mu[k - 1])
    for n in range(1,N + 1):
        factor(f'expr__{n}__2', log_sum_exp_array(soft_z[n - 1]))


def generated_quantities(__inputs__):
    K = __inputs__['K']
    N = __inputs__['N']
    y = __inputs__['y']
    neg_log_K = __inputs__['neg_log_K']
    mu = __inputs__['mu']
    # Transformed parameters
    soft_z = empty([N, K], dtype=dtype_float)
    for n in range(1,N + 1):
        for k in range(1,K + 1):
            soft_z[n - 1, k - 1] = neg_log_K - array(0.5, dtype=dtype_float) * dot_self_vector(
            mu[k - 1] - y[n - 1])
    return { 'soft_z': soft_z }
