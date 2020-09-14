from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def transformed_data(*, N, D, K, y):
    # Transformed data
    neg_log_K = None
    neg_log_K = - log(K)
    return { 'neg_log_K': neg_log_K }

def model(*, N, D, K, y, neg_log_K):
    # Parameters
    mu = sample('mu', improper_uniform(shape=[K, D]))
    # Transformed parameters
    soft_z = zeros([N, K])
    for n in range(1,N + 1):
        for k in range(1,K + 1):
            soft_z[n - 1][k - 1] = neg_log_K - 0.5 * dot_self(mu[k - 1] - y[n - 1])
    # Model
    for k in range(1,K + 1):
        observe(f'mu__{k}__1', normal(0, 1), mu[k - 1])
    for n in range(1,N + 1):
        factor(f'expr__{n}__2', log_sum_exp(soft_z[n - 1]))


def generated_quantities(*, N, D, K, y, neg_log_K, mu):
    # Transformed parameters
    soft_z = zeros([N, K])
    for n in range(1,N + 1):
        for k in range(1,K + 1):
            soft_z[n - 1][k - 1] = neg_log_K - 0.5 * dot_self(mu[k - 1] - y[n - 1])
    return { 'soft_z': soft_z }
