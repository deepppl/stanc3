def transformed_data(N=None, exposure2=None, roach1=None, senior=None,
    treatment=None, y=None):
    log_expo = zeros(N)
    sqrt_roach = zeros(N)
    log_expo = log(exposure2)
    sqrt_roach = sqrt(roach1)
    return {'log_expo': log_expo, 'sqrt_roach': sqrt_roach}


def model(N=None, exposure2=None, roach1=None, senior=None, treatment=None,
    y=None, transformed_data=None):
    sqrt_roach = transformed_data['sqrt_roach']
    log_expo = transformed_data['log_expo']
    beta_1 = sample('beta_1', dist.Normal(0, 5))
    beta_2 = sample('beta_2', dist.Normal(0, 2.5))
    beta_3 = sample('beta_3', dist.Normal(0, 2.5))
    beta_4 = sample('beta_4', dist.Normal(0, 2.5))
    sample('y', poisson_log(log_expo + beta_1 + beta_2 *
        sqrt_roach + beta_3 * treatment + beta_4 * senior), obs=y)
