
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(D=None, K=None, N=None, y=None):
    neg_log_K = -log(K)
    return {'neg_log_K': neg_log_K}


def model(D=None, K=None, N=None, y=None, transformed_data=None):
    neg_log_K = transformed_data['neg_log_K']
    mu = sample('mu', ImproperUniform(shape=(K, D)))
    soft_z = zeros((N, K))
    for n in range(1, N + 1):
        for k in range(1, K + 1):
            soft_z[n - 1, k - 1] = neg_log_K - 0.5 * dot_self(mu[k - 1] - y
                [n - 1])
    for k in range(1, K + 1):
        sample('mu' + '__{}'.format(k - 1) + '__1', dist.Normal(zeros(D), 1
            ), obs=mu[k - 1])
    for n in range(1, N + 1):
        sample('expr' + '__{}'.format(n) + '__2', dist.Exponential(1.0),
            obs=-log_sum_exp(soft_z[n - 1]))


def generated_quantities(D=None, K=None, N=None, y=None, transformed_data=
    None, parameters=None):
    neg_log_K = transformed_data['neg_log_K']
    mu = parameters['mu']
    soft_z = zeros((N, K))
    for n in range(1, N + 1):
        for k in range(1, K + 1):
            soft_z[n - 1, k - 1] = neg_log_K - 0.5 * dot_self(mu[k - 1] - y
                [n - 1])
    return {'soft_z': soft_z}