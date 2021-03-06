from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap
from stanpyro.stanlib import pow_real_int, pow_real_real

def convert_inputs(inputs):
    N = inputs['N']
    y = array(inputs['y'], dtype=dtype_float)
    s = array(inputs['s'], dtype=dtype_float)
    mu_loc = array(inputs['mu_loc'], dtype=dtype_float)
    mu_scale = array(inputs['mu_scale'], dtype=dtype_float)
    tau_scale = array(inputs['tau_scale'], dtype=dtype_float)
    tau_df = array(inputs['tau_df'], dtype=dtype_float)
    return { 'N': N, 'y': y, 's': s, 'mu_loc': mu_loc, 'mu_scale': mu_scale,
             'tau_scale': tau_scale, 'tau_df': tau_df }

def model(*, N, y, s, mu_loc, mu_scale, tau_scale, tau_df):
    # Parameters
    theta_raw = sample('theta_raw', improper_uniform(shape=[N]))
    mu = sample('mu', improper_uniform(shape=[]))
    tau = sample('tau', improper_uniform(shape=[]))
    # Transformed parameters
    theta = tau * theta_raw + mu
    # Model
    observe('_mu__1', normal(mu_loc, mu_scale), mu)
    observe('_tau__2', student_t(tau_df, array(0., dtype=dtype_float),
                                 tau_scale), tau)
    observe('_theta_raw__3', normal(array(0., dtype=dtype_float),
                                    array(1., dtype=dtype_float)), theta_raw)
    observe('_y__4', normal(theta, s), y)


def generated_quantities(*, N, y, s, mu_loc, mu_scale, tau_scale, tau_df,
                            theta_raw, mu, tau):
    # Transformed parameters
    theta = tau * theta_raw + mu
    # Generated quantities
    tau2 = pow_real_real(tau, array(2., dtype=dtype_float))
    shrinkage = empty([N], dtype=dtype_float)
    for i in range(1,N + 1):
        v = pow_real_int(s[i - 1], 2)
        shrinkage[i - 1] = true_divide(v, (v + tau2))
    return { 'theta': theta, 'tau2': tau2, 'shrinkage': shrinkage }

def map_generated_quantities(_samples, *, N, y, s, mu_loc, mu_scale,
                                          tau_scale, tau_df):
    def _generated_quantities(theta_raw, mu, tau):
        return generated_quantities(N=N, y=y, s=s, mu_loc=mu_loc,
                                    mu_scale=mu_scale, tau_scale=tau_scale,
                                    tau_df=tau_df, theta_raw=theta_raw,
                                    mu=mu, tau=tau)
    _f = vmap(_generated_quantities)
    return _f(_samples['theta_raw'], _samples['mu'], _samples['tau'])
