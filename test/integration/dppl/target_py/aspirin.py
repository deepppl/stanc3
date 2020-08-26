from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, observe, factor, array, zeros, ones
from runtimes.pyro.stanlib import sqrt, exp, log

def model(*, N, y, s, mu_loc, mu_scale, tau_scale, tau_df):
    # Parameters
    theta_raw = sample('theta_raw', improper_uniform(shape=[N]))
    mu = sample('mu', improper_uniform(shape=None))
    tau = sample('tau', improper_uniform(shape=None))
    # Transformed parameters
    theta = zeros([N])
    theta = tau * theta_raw + mu
    # Model
    observe('mu__1', normal(mu_loc, mu_scale), mu)
    observe('tau__2', student_t(tau_df, 0., tau_scale), tau)
    observe('theta_raw__3', normal(0., 1.), theta_raw)
    observe('y__4', normal(theta, s), y)

def generated_quantities(*, N, y, s, mu_loc, mu_scale, tau_scale, tau_df,
                            theta_raw, mu, tau):
    # Transformed parameters
    theta = zeros([N])
    theta = tau * theta_raw + mu
    # Generated quantities
    shrinkage = zeros([N])
    tau2 = None
    tau2 = pow(tau, 2.)
    for i in range(1,N + 1):
        v = None
        v = pow(s[i - 1], 2)
        shrinkage[i - 1] = v / (v + tau2)
    return { 'theta': theta, 'shrinkage': shrinkage, 'tau2': tau2 }
