
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(x: 'int[10]'=None):
    theta: 'real' = sample('theta', dist.Uniform(0.0, 1.0))
    sample('theta' + '__1', dist.Uniform(0 * 3 / 5, 1 + 5 - 5), obs=theta)
    for i in range(1, 10 + 1):
        if 1 <= 10 and (1 > 5 or 2 < 1):
            sample('x' + '__{}'.format(i - 1) + '__2', dist.Bernoulli(
                theta), obs=x[i - 1])
