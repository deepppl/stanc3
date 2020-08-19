
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(x=None):
    ___shape = {}
    ___shape['y'] = 10
    y = zeros(___shape['y'])
    for i in range(1, 10 + 1):
        y[i - 1] = 1 - x[i - 1]
    return {'y': y}


def model(x=None, transformed_data=None):
    y = transformed_data['y']
    ___shape = {}
    ___shape['x'] = 10
    ___shape['theta'] = ()
    theta = sample('theta', dist.Uniform(0.0, 1.0))
    sample('theta' + '__1', dist.Uniform(0, 1), obs=theta)
    for i in range(1, 10 + 1):
        sample('y' + '__{}'.format(i - 1) + '__2', dist.Bernoulli(theta),
            obs=y[i - 1])
