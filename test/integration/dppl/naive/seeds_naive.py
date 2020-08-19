def transformed_data(I=None, N=None, n=None, x1=None, x2=None):
    x1x2 = zeros(I)
    x1x2 = x1 * x2
    return {'x1x2': x1x2}


def model(I=None, N=None, n=None, x1=None, x2=None, transformed_data=None):
    x1x2 = transformed_data['x1x2']
    alpha0 = sample('alpha0', dist.Normal(0.0, 1000))
    alpha1 = sample('alpha1', dist.Normal(0.0, 1000))
    alpha2 = sample('alpha2', dist.Normal(0.0, 1000))
    alpha12 = sample('alpha12', dist.Normal(0.0, 1000))
    tau = sample('tau', dist.Gamma(0.001, 0.001))
    sigma = 1.0 / sqrt(tau)
    b = sample('b', dist.Normal(zeros(I), sigma))
    sample('n', binomial_logit(N, alpha0 + alpha1 * x1 + alpha2 *
        x2 + alpha12 * x1x2 + b), obs=n)


def generated_quantities(I=None, N=None, n=None, x1=None, x2=None,
    transformed_data=None, parameters=None):
    x1x2 = transformed_data['x1x2']
    alpha0 = parameters['alpha0']
    alpha1 = parameters['alpha1']
    alpha12 = parameters['alpha12']
    alpha2 = parameters['alpha2']
    b = parameters['b']
    tau = parameters['tau']
    sigma = 1.0 / sqrt(tau)
    return {'sigma': sigma}