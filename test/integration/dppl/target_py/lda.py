

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(K=None, M=None, N=None, V=None, alpha=None, beta=None, doc=None,
    w=None):
    theta = sample('theta', ImproperUniform(shape=(M, K)))
    phi = sample('phi', ImproperUniform(shape=(K, V)))
    for m in range(1, M + 1):
        sample('theta' + '__{}'.format(m - 1) + '__1', dist.Dirichlet(
            alpha), obs=theta[m - 1])
    for k in range(1, K + 1):
        sample('phi' + '__{}'.format(k - 1) + '__2', dist.Dirichlet(
            beta), obs=phi[k - 1])
    for n in range(1, N + 1):
        gamma = zeros(K)
        for k in range(1, K + 1):
            gamma[k - 1] = log(theta[doc[n - 1] - 1, k - 1]) + log(phi[k -
                1, w[n - 1] - 1])
        sample('expr' + '__{}'.format(n) + '__3', dist.Exponential(1.0
            ), obs=-log_sum_exp(gamma))
