from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap
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
        observe(f'_mu__{k}__1', normal(0, 1), mu[k - 1])
    for n in range(1,N + 1):
        factor(f'_expr__{n}__2', log_sum_exp_array(soft_z[n - 1]))


def generated_quantities(*, K, N, D, y, neg_log_K, mu):
    # Transformed parameters
    soft_z = empty([N, K], dtype=dtype_float)
    for n in range(1,N + 1):
        for k in range(1,K + 1):
            soft_z[n - 1, k - 1] = neg_log_K - array(0.5, dtype=dtype_float) * dot_self_vector(
            mu[k - 1] - y[n - 1])
    return { 'soft_z': soft_z }

def map_generated_quantities(_samples, *, K, N, D, y, neg_log_K):
    def _generated_quantities(mu):
        return generated_quantities(K=K, N=N, D=D, y=y, neg_log_K=neg_log_K,
                                    mu=mu)
    return vmap(_generated_quantities)(_samples['mu'])
