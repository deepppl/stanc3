

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(K=None, N=None, x=None, y=None):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['K'] = ()
    ___shape['x'] = N, K
    ___shape['y'] = N
    ___shape['alpha'] = ()
    ___shape['beta'] = K
    ___shape['sigma'] = ()
    alpha = sample('alpha', ImproperUniform())
    beta = sample('beta', ImproperUniform(K))
    sigma = sample('sigma', LowerConstrainedImproperUniform(0.0))
    sample('y' + '__1', dist.Normal(x * beta + alpha, sigma), obs=y)
