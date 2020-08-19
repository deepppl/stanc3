

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(N=None, x=None):
    ___shape = {}
    ___shape['K'] = N, N
    K = zeros(___shape['K'])
    ___shape['mu'] = N
    mu = rep_vector(0, N)
    for i in range(1, N - 1 + 1):
        K[i - 1, i - 1] = 1 + 0.1
        for j in range(i + 1, N + 1):
            K[i - 1, j - 1] = exp(-0.5 * square(x[i - 1] - x[j - 1]))
            K[j - 1, i - 1] = K[i - 1, j - 1]
    K[N - 1, N - 1] = 1 + 0.1
    return {'K': K, 'mu': mu}


def model(N=None, x=None, transformed_data=None):
    K = transformed_data['K']
    mu = transformed_data['mu']
    ___shape = {}
    ___shape['N'] = ()
    ___shape['x'] = N
    ___shape['y'] = N
    y = sample('y', ImproperUniform(N))
    sample('y' + '__1', dist.MultivariateNormal(mu, K), obs=y)
