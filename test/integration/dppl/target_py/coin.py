from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N, x):
    # Parameters
    z = sample('z', uniform(0, 1))
    # Model
    observe('z__1', beta(1, 1), z)
    for i in range(1,N + 1):
        observe(f'x__{i}__2', bernoulli(z), x[i - 1])

