
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(x: 'int[10]'=None):
    theta: 'real' = sample('theta', dist.Uniform(0.0, 1.0))
    sample('theta' + '__1', dist.Uniform(0.0, 1.0), obs=theta)
    sample('x' + '__2', dist.Bernoulli(theta * ones(10)), obs=x)
