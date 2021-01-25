import pyro
from pyro.distributions import Exponential, Bernoulli, Binomial, Poisson, GammaPoisson
from pyro import module as register_network
from pyro import random_module
from torch import (
    tensor,
    zeros,
    ones,
    Tensor,
    matmul,
    true_divide,
    floor_divide,
    transpose,
    empty,
    stack,
)
from torch import long as dtype_long
from torch import float as dtype_float
from collections import defaultdict

import torch

def _cuda(f):
    def inner(*args, **kwargs):
        return f(*args, **kwargs).cuda()

    return inner


zeros = _cuda(zeros)
ones = _cuda(ones)
tensor = _cuda(tensor)
empty = _cuda(empty)

def array(x, dtype=None):
  if isinstance(x, list):
    return stack([tensor(e, dtype=dtype) for e in x])
  return tensor(x, dtype=dtype)

def vmap(f):
    def vmap(*args):
        n = len(args[0])
        res = defaultdict(list)
        for i in range(n):
            d = f(*[ x[i] for x in args ])
            for k, v in d.items():
                res[k].append(v)
        return {k: stack(v) for k, v in res.items()}
    return vmap

def sample(site_name, dist, *args, **kwargs):
    return pyro.sample(site_name, dist, *args, **kwargs)


def param(site_name, init):
    return pyro.param(site_name, init)


def observe(site_name, dist, obs):
    if isinstance(dist, (Bernoulli, Binomial, Poisson, GammaPoisson)):
        obs = (
            obs.type(dtype_float)
            if isinstance(obs, dtype_long)
            else array(obs, dtype=dtype_float)
        )
    pyro.sample(site_name, dist, obs=obs)


def factor(site_name, x):
    pyro.factor(site_name, x)
