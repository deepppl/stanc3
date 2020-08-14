import numpyro
from numpyro.distributions import Exponential
from numpy import array

def sample(site_name, dist):
    return numpyro.sample(site_name, dist)

def observe(site_name, dist, obs):
    numpyro.sample(site_name, dist, obs = obs)

def factor(site_name, x):
    numpyro.sample(site_name, Exponential(1), obs=-x)
