

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_():
    mu_cluster = pyro.param('mu_cluster', (2 - -2) * rand(()) + -2)
    mu1 = pyro.param('mu1', (2 - -2) * rand(()) + -2)
    mu2 = pyro.param('mu2', (2 - -2) * rand(()) + -2)
    log_sigma1 = pyro.param('log_sigma1', (2 - -2) * rand(()) + -2)
    log_sigma2 = pyro.param('log_sigma2', (2 - -2) * rand(()) + -2)
    cluster = sample('cluster', dist.Normal(mu_cluster, 1))
    if cluster > 0:
        theta = sample('theta', dist.Normal(mu1, exp(log_sigma1)))
    else:
        theta = sample('theta', dist.Normal(mu2, exp(log_sigma2)))


def model():
    cluster = sample('cluster', ImproperUniform())
    theta = sample('theta', ImproperUniform())
    sample('cluster' + '__1', dist.Normal(0, 1), obs=cluster)
    if cluster > 0:
        mu = 2
    else:
        mu = 0
    sample('theta' + '__2', dist.Normal(mu, 1), obs=theta)
