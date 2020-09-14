from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N, y, sigma_y):
    # Parameters
    eta = sample('eta', improper_uniform(shape=[N]))
    mu_theta = sample('mu_theta', improper_uniform(shape=None))
    sigma_eta = sample('sigma_eta', uniform(0, 100))
    xi = sample('xi', improper_uniform(shape=None))
    # Transformed parameters
    theta = zeros([N])
    theta = mu_theta + xi * eta
    # Model
    observe('mu_theta__1', normal(0, 100), mu_theta)
    observe('sigma_eta__2', inv_gamma(1, 1), sigma_eta)
    observe('eta__3', normal(0, sigma_eta), eta)
    observe('xi__4', normal(0, 5), xi)
    observe('y__5', normal(theta, sigma_y), y)


def generated_quantities(*, N, y, sigma_y, eta, mu_theta, sigma_eta, xi):
    # Transformed parameters
    theta = zeros([N])
    theta = mu_theta + xi * eta
    return { 'theta': theta }
