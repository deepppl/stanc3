from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

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
    observe('_mu_theta__1', normal(0, 100), mu_theta)
    observe('_sigma_eta__2', inv_gamma(1, 1), sigma_eta)
    observe('_eta__3', normal(0, sigma_eta), eta)
    observe('_xi__4', normal(0, 5), xi)
    observe('_y__5', normal(theta, sigma_y), y)


def generated_quantities(*, N, y, sigma_y, eta, mu_theta, sigma_eta, xi):
    # Transformed parameters
    theta = mu_theta + xi * eta
    return { 'theta': theta }

def map_generated_quantities(_samples, *, N, y, sigma_y):
    def _generated_quantities(eta, mu_theta, sigma_eta, xi):
        return generated_quantities(N=N, y=y, sigma_y=sigma_y, eta=eta,
                                    mu_theta=mu_theta, sigma_eta=sigma_eta,
                                    xi=xi)
    return vmap(_generated_quantities)(_samples['eta'], _samples['mu_theta'],
                                       _samples['sigma_eta'], _samples['xi'])
