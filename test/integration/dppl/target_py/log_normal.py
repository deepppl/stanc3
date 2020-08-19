

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    theta: 'real' = sample('theta', LowerConstrainedImproperUniform(0.0))
    sample('expr' + '__1', dist.Normal(log(10.0), 1.0), obs=log(theta))
    sample('expr' + '__2', dist.Exponential(1.0), obs=--log(fabs(theta)))
