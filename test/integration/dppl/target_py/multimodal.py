
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    cluster = sample('cluster', ImproperUniform())
    theta = sample('theta', ImproperUniform())
    sample('cluster' + '__1', dist.Normal(0, 1), obs=cluster)
    if cluster > 0:
        mu = 2
    else:
        mu = 0
    sample('theta' + '__2', dist.Normal(mu, 1), obs=theta)
