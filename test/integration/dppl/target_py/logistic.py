

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(M=None, N=None, x=None, y=None):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['M'] = ()
    ___shape['y'] = N
    ___shape['x'] = N, M
    ___shape['beta'] = M
    beta = sample('beta', ImproperUniform(M))
    for m in range(1, M + 1):
        sample('beta' + '__{}'.format(m - 1) + '__1', dist.Cauchy(0.0, 2.5
            ), obs=beta[m - 1])
    for n in range(1, N + 1):
        sample('y' + '__{}'.format(n - 1) + '__2', dist.Bernoulli(
            inv_logit(x[n - 1] * beta)), obs=y[n - 1])
