
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N_mis: 'int'=None, N_obs: 'int'=None, y_obs: 'real[N_obs]'=None):
    mu: 'real' = sample('mu', ImproperUniform())
    sigma: 'real' = sample('sigma', LowerConstrainedImproperUniform(0.0))
    y_mis: 'real[N_mis]' = sample('y_mis', ImproperUniform(shape=N_mis))
    sample('y_obs' + '__1', dist.Normal(mu * ones(N_obs), sigma), obs=
        y_obs)
    sample('y_mis' + '__2', dist.Normal(mu * ones(N_mis), sigma), obs=
        y_mis)