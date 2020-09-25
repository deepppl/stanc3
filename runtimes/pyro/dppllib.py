import pyro
from pyro.distributions import Exponential
from torch import tensor as array
from torch import zeros, ones, Tensor, matmul, true_divide, floor_divide
import torch

def sample(site_name, dist):
    return pyro.sample(site_name, dist)

def param(site_name, init):
    return pyro.param(site_name, init)

def observe(site_name, dist, obs):
    pyro.sample(site_name, dist, obs = obs)

def factor(site_name, x):
    pyro.sample(site_name, Exponential(1), obs=-x)

def register_network(name, x):
    pyro.module(name, x)

dtype_long = torch.long
dtype_float = torch.float
