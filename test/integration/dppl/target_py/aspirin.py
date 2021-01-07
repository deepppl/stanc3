from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float
from runtimes.pyro.stanlib import pow_real_int, pow_real_real

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
    observe('mu__1', normal(mu_loc, mu_scale), mu)
    observe('tau__2', student_t(tau_df, array(0., dtype=dtype_float),
                                tau_scale), tau)
    observe('theta_raw__3', normal(array(0., dtype=dtype_float),
                                   array(1., dtype=dtype_float)), theta_raw)
    observe('y__4', normal(theta, s), y)


def generated_quantities(__inputs__):
    N = __inputs__['N']
    s = __inputs__['s']
    theta_raw = __inputs__['theta_raw']
    mu = __inputs__['mu']
    tau = __inputs__['tau']
    # Transformed parameters
    theta = tau * theta_raw + mu
    # Generated quantities
    tau2 = pow_real_real(tau, array(2., dtype=dtype_float))
    shrinkage = empty([N], dtype=dtype_float)
    for i in range(1,N + 1):
        v = pow_real_int(s[i - 1], 2)
        shrinkage[i - 1] = true_divide(v, (v + tau2))
    return { 'theta': theta, 'tau2': tau2, 'shrinkage': shrinkage }
