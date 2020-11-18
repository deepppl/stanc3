import pyro
from pyro.distributions import Exponential, Bernoulli, Binomial, Poisson, GammaPoisson
from pyro import module as register_network
from pyro import random_module
from torch import tensor, zeros, ones, Tensor, matmul, true_divide, floor_divide, transpose, empty
from torch import long as dtype_long
from torch import float as dtype_float

import torch

def _cuda(f):
    def inner(*args, **kwargs):
        return f(*args, **kwargs).cuda()
    return inner

zeros = _cuda(zeros)
ones = _cuda(ones)
array = _cuda(tensor)
empty = _cuda(empty)

def sample(site_name, dist, *args, **kwargs):
    return pyro.sample(site_name, dist, *args, **kwargs)

def param(site_name, init):
    return pyro.param(site_name, init)

def observe(site_name, dist, obs):
    if isinstance(dist, (Bernoulli, Binomial, Poisson, GammaPoisson)):
        obs = obs.type(dtype_float) if isinstance(obs, dtype_long) else array(obs, dtype=dtype_float)
    pyro.sample(site_name, dist, obs = obs)

def factor(site_name, x):
    pyro.factor(site_name, x)
