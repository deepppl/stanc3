
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(I=None, N=None, n=None, x1=None, x2=None):
    x1x2 = zeros(I)
    x1x2 = x1 * x2
    return {'x1x2': x1x2}


def model(I=None, N=None, n=None, x1=None, x2=None, transformed_data=None):
    x1x2 = transformed_data['x1x2']
    alpha0 = sample('alpha0', ImproperUniform())
    alpha1 = sample('alpha1', ImproperUniform())
    alpha12 = sample('alpha12', ImproperUniform())
    alpha2 = sample('alpha2', ImproperUniform())
    tau = sample('tau', LowerConstrainedImproperUniform(0.0))
    b = sample('b', ImproperUniform(shape=I))
    sigma = 1.0 / sqrt(tau)
    sample('alpha0' + '__1', dist.Normal(0.0, 1000), obs=alpha0)
    sample('alpha1' + '__2', dist.Normal(0.0, 1000), obs=alpha1)
    sample('alpha2' + '__3', dist.Normal(0.0, 1000), obs=alpha2)
    sample('alpha12' + '__4', dist.Normal(0.0, 1000), obs=alpha12)
    sample('tau' + '__5', dist.Gamma(0.001, 0.001), obs=tau)
    sample('b' + '__6', dist.Normal(zeros(I), sigma), obs=b)
    sample('n' + '__7', binomial_logit(N, alpha0 + alpha1 * x1 + alpha2 *
        x2 + alpha12 * x1x2 + b), obs=n)


def generated_quantities(I=None, N=None, n=None, x1=None, x2=None,
    transformed_data=None, parameters=None):
    x1x2 = transformed_data['x1x2']
    alpha0 = parameters['alpha0']
    alpha1 = parameters['alpha1']
    alpha12 = parameters['alpha12']
    alpha2 = parameters['alpha2']
    b = parameters['b']
    tau = parameters['tau']
    sigma = 1.0 / sqrt(tau)
    return {'sigma': sigma}