
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(x):
    theta = sample('theta', dist.Uniform(torch(0.0), torch(1.0)))
    sample('x', dist.Bernoulli(theta), obs=x)
