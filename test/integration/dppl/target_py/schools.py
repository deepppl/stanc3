from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network

def convert_inputs(inputs):
    N = inputs['N']
    y = array(inputs['y'], dtype=dtype_float)
    sigma_y = array(inputs['sigma_y'], dtype=dtype_float)
    return { 'N': N, 'y': y, 'sigma_y': sigma_y }

def model(*, N, y, sigma_y):
    # Parameters
    eta = sample('eta', improper_uniform(shape=[N]))
    mu_theta = sample('mu_theta', improper_uniform(shape=[]))
    sigma_eta = sample('sigma_eta', uniform(0, 100))
    xi = sample('xi', improper_uniform(shape=[]))
    # Transformed parameters
    theta = mu_theta + xi * eta
    # Model
    observe('mu_theta__1', normal(0, 100), mu_theta)
    observe('sigma_eta__2', inv_gamma(1, 1), sigma_eta)
    observe('eta__3', normal(0, sigma_eta), eta)
    observe('xi__4', normal(0, 5), xi)
    observe('y__5', normal(theta, sigma_y), y)


def generated_quantities(__inputs__):
    N = __inputs__['N']
    y = __inputs__['y']
    sigma_y = __inputs__['sigma_y']
    eta = __inputs__['eta']
    mu_theta = __inputs__['mu_theta']
    sigma_eta = __inputs__['sigma_eta']
    xi = __inputs__['xi']
    # Transformed parameters
    theta = mu_theta + xi * eta
    return { 'theta': theta }
