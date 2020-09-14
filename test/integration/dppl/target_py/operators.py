from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, x):
    # Parameters
    theta = sample('theta', uniform(0, 1))
    # Model
    observe('theta__1', uniform(0 * 3 / 5, 1 + 5 - 5), theta)
    for i in range(1,10 + 1):
        if (1 <= 10) and (1 > 5 or 2 < 1):
            observe(f'x__{i}__2', bernoulli(theta), x[i - 1])

