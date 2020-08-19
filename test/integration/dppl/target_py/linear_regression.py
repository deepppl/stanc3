
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N: 'int'=None, x: 'real[N]'=None, y: 'real[N]'=None):
    alpha: 'real' = sample('alpha', ImproperUniform())
    beta: 'real' = sample('beta', ImproperUniform())
    sigma: 'real' = sample('sigma', LowerConstrainedImproperUniform(0.0))
    sample('y' + '__1', dist.Normal(alpha + beta * x, sigma), obs=y)
