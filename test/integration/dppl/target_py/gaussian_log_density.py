
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    theta: 'real' = sample('theta', ImproperUniform())
    sample('expr' + '__1', dist.Exponential(1.0), 
        obs=-(-0.5 * (theta - 1000.0) * (theta - 1000.0)))
