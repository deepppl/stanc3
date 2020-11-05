from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float
from runtimes.pyro.stanlib import log_real, log_sum_exp_array

def convert_inputs(inputs):
    N = inputs['N']
    V = inputs['V']
    w = array(inputs['w'], dtype=dtype_long)
    M = inputs['M']
    doc = array(inputs['doc'], dtype=dtype_long)
    K = inputs['K']
    alpha = array(inputs['alpha'], dtype=dtype_float)
    beta__ = array(inputs['beta'], dtype=dtype_float)
    return { 'N': N, 'V': V, 'w': w, 'M': M, 'doc': doc, 'K': K,
             'alpha': alpha, 'beta__': beta__ }

def model(*, N, V, w, M, doc, K, alpha, beta__):
    # Parameters
    theta = sample('theta', simplex_constrained_improper_uniform(shape=[
    M, K]))
    phi = sample('phi', simplex_constrained_improper_uniform(shape=[K, V]))
    # Model
    for m in range(1,M + 1):
        observe(f'theta__{m}__1', dirichlet(alpha), theta[m - 1])
    for k in range(1,K + 1):
        observe(f'phi__{k}__2', dirichlet(beta__), phi[k - 1])
    for n in range(1,N + 1):
        gamma__ = empty([K], dtype=dtype_float)
        for k in range(1,K + 1):
            gamma__[k - 1] = log_real(theta[doc[n - 1] - 1, k - 1]) + log_real(
            phi[k - 1, w[n - 1] - 1])
        factor(f'expr__{n}__3', log_sum_exp_array(gamma__.clone()))

