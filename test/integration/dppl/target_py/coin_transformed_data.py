from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def transformed_data(*, x):
    # Transformed data
    y = zeros([10])
    for i in range(1,10 + 1):
        y[i - 1] = 1 - x[i - 1]
    return { 'y': y }

def model(*, x, y):
    # Parameters
    theta = sample('theta', uniform(0, 1))
    # Model
    observe('theta__1', uniform(0, 1), theta)
    for i in range(1,10 + 1):
        observe(f'y__{i}__2', bernoulli(theta), y[i - 1])

