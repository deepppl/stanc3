def model(N=None, mu_loc=None, mu_scale=None, s=None, tau_df=None,
    tau_scale=None, y=None):
    mu = sample('mu', dist.Normal(mu_loc, mu_loc))
    tau = sample('tau', dist.StudentT(tau_df, 0.0, tau_scale))
    theta = zeros(N)
    theta_raw = sample('theta_raw', dist.Normal(zeros(N), 1.0))
    theta = tau * theta_raw + mu
    sample('y', dist.Normal(theta, s), obs=y)


def generated_quantities(N=None, mu_loc=None, mu_scale=None, s=None, tau_df
    =None, tau_scale=None, y=None, parameters=None):
    mu = parameters['mu']
    tau = parameters['tau']
    theta_raw = parameters['theta_raw']
    theta = zeros(N)
    theta = tau * theta_raw + mu
    shrinkage = zeros(N)
    tau2 = pow(tau, 2.0)
    for i in range(1, N + 1):
        v = pow(s[i - 1], 2)
        shrinkage[i - 1] = v / (v + tau2)
    return {'theta': theta, 'shrinkage': shrinkage, 'tau2': tau2}