import pyro
from pyro.distributions import Exponential, Bernoulli, Binomial, Poisson, GammaPoisson
from torch import tensor as array
from torch import zeros, ones, Tensor, matmul, true_divide, floor_divide, transpose, empty
from torch import LongTensor
from torch import long as dtype_long
from torch import float as dtype_float

import torch

def sample(site_name, dist, *args, **kwargs):
    return pyro.sample(site_name, dist, *args, **kwargs)

def param(site_name, init):
    return pyro.param(site_name, init)

def observe(site_name, dist, obs):
    if isinstance(dist, (Bernoulli, Binomial, Poisson, GammaPoisson)):
        obs = obs.type(dtype_float) if isinstance(obs, LongTensor) else array(obs, dtype=dtype_float)
    pyro.sample(site_name, dist, obs = obs)

def factor(site_name, x):
    pyro.sample(site_name, Exponential(1), obs=-x)

def register_network(name, x):
    pyro.module(name, x)
