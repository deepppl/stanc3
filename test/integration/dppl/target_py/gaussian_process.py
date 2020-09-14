from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def transformed_data(*, N, x):
    # Transformed data
    K = zeros([N, N])
    mu = rep_vector(0, N)
    for i in range(1,(N - 1) + 1):
        K[i - 1][i - 1] = 1 + 0.1
        for j in range((i + 1),N + 1):
            K[i - 1][j - 1] = exp(- 0.5 * square(x[i - 1] - x[j - 1]))
            K[j - 1][i - 1] = K[i - 1][j - 1]
    K[N - 1][N - 1] = 1 + 0.1
    return { 'K': K, 'mu': mu }

def model(*, N, x, K, mu):
    # Parameters
    y = sample('y', improper_uniform(shape=[N]))
    # Model
    observe('y__1', multi_normal(mu, K), y)

