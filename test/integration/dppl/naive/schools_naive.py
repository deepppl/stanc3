def model(N=None, sigma_y=None, y=None):
    theta = zeros(N)
    mu_theta = sample('mu_theta', dist.Normal(0, 100))
    sigma_eta = sample('sigma_eta', dist.InverseGamma(1, 1))
    eta = sample('eta', dist.Normal(zeros(N), sigma_eta))
    xi = sample('xi', dist.Normal(0, 5))
    theta = mu_theta + xi * eta
    sample('y', dist.Normal(theta, sigma_y), obs=y)


def generated_quantities(N=None, sigma_y=None, y=None, parameters=None):
    eta = parameters['eta']
    mu_theta = parameters['mu_theta']
    sigma_eta = parameters['sigma_eta']
    xi = parameters['xi']
    theta = zeros(N)
    theta = mu_theta + xi * eta
    return {'theta': theta}
