

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    y_std: 'real' = sample('y_std', ImproperUniform())
    x_std: 'real' = sample('x_std', ImproperUniform())
    y: 'real' = 3.0 * y_std
    x: 'real' = exp(y / 2) * x_std
    sample('y_std' + '__1', dist.Normal(0, 1), obs=y_std)
    sample('x_std' + '__2', dist.Normal(0, 1), obs=x_std)


def generated_quantities(parameters=None):
    x_std = parameters['x_std']
    y_std = parameters['y_std']
    y: 'real' = 3.0 * y_std
    x: 'real' = exp(y / 2) * x_std
    return {'x': x, 'y': y}
