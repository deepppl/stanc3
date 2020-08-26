from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def transformed_data(*, I, n, N, x1, x2):
    # Transformed data
    x1x2 = zeros([I])
    x1x2 = x1 * x2
    return { 'x1x2': x1x2 }

def model(*, I, n, N, x1, x2, x1x2):
    # Parameters
    alpha0 = sample('alpha0', improper_uniform(shape=None))
    alpha1 = sample('alpha1', improper_uniform(shape=None))
    alpha12 = sample('alpha12', improper_uniform(shape=None))
    alpha2 = sample('alpha2', improper_uniform(shape=None))
    tau = sample('tau', lower_constrained_improper_uniform(0, shape=None))
    b = sample('b', improper_uniform(shape=[I]))
    # Transformed parameters
    sigma = 1.0 / sqrt(tau)
    # Model
    observe('alpha0__1', normal(0.0, 1000), alpha0)
    observe('alpha1__2', normal(0.0, 1000), alpha1)
    observe('alpha2__3', normal(0.0, 1000), alpha2)
    observe('alpha12__4', normal(0.0, 1000), alpha12)
    observe('tau__5', gamma(0.001, 0.001), tau)
    observe('b__6', normal(0.0, sigma), b)
    observe('n__7', binomial_logit(N,
                                   alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b), n)

def generated_quantities(*, I, n, N, x1, x2, x1x2, alpha0, alpha1, alpha12,
                            alpha2, tau, b):
    # Transformed parameters
    sigma = 1.0 / sqrt(tau)
    return { 'sigma': sigma }
