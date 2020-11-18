import numpyro as pyro
import jax.numpy as tensor
from numpyro.distributions import Exponential
from pyro import module as register_network
from pyro import random_module
from jax.numpy import array
from jax.numpy import zeros, ones, matmul, true_divide, floor_divide, transpose, empty
dtype_float=tensor.dtype('float32')
dtype_long=tensor.dtype('int32')

def sample(site_name, dist, *args, **kwargs):
    return pyro.sample(site_name, dist, *args, **kwargs)

def param(site_name, init):
    return pyro.param(site_name, init)

def observe(site_name, dist, obs):
    pyro.sample(site_name, dist, obs = obs)

def factor(site_name, x):
    pyro.factor(site_name, x)

from jax.ops import index as ops_index
from jax.ops import index_update as ops_index_update
from jax.lax import cond as lax_cond
from jax.lax import while_loop as lax_while_loop
from jax.lax import fori_loop as lax_fori_loop
from jax.lax import map as lax_map
