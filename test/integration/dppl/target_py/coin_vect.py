
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N=None, x=None):
    z = sample('z', dist.Uniform(0.0, 1.0))
    sample('z' + '__1', dist.Beta(1, 1), obs=z)
    sample('x' + '__2', dist.Bernoulli(z * ones(N)), obs=x)
