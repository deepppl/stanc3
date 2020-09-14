from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N, M, y, x):
    # Parameters
    beta = sample('beta', improper_uniform(shape=[M]))
    # Model
    for m in range(1,M + 1):
        observe(f'beta__{m}__1', cauchy(0.0, 2.5), beta[m - 1])
    for n in range(1,N + 1):
        observe(f'y__{n}__2', bernoulli(inv_logit(dot(x[n - 1], beta))), y[n - 1])


