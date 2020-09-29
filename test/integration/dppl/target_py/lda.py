from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network
from runtimes.pyro.stanlib import log_real, log_sum_exp_array

def convert_inputs(inputs):
    K = inputs['K']
    V = inputs['V']
    M = inputs['M']
    N = inputs['N']
    w = array(inputs['w'], dtype=dtype_long)
    doc = array(inputs['doc'], dtype=dtype_long)
    alpha = array(inputs['alpha'], dtype=dtype_float)
    beta = array(inputs['beta'], dtype=dtype_float)
    return { 'K': K, 'V': V, 'M': M, 'N': N, 'w': w, 'doc': doc,
             'alpha': alpha, 'beta': beta }

def model(*, K, V, M, N, w, doc, alpha, beta):
    # Parameters
    theta = sample('theta', simplex_constrained_improper_uniform(shape=[
    M, K]))
    phi = sample('phi', simplex_constrained_improper_uniform(shape=[K, V]))
    # Model
    for m in range(1,M + 1):
        observe(f'theta__{m}__1', dirichlet(alpha), theta[m - 1])
    for k in range(1,K + 1):
        observe(f'phi__{k}__2', dirichlet(beta), phi[k - 1])
    for n in range(1,N + 1):
        gamma = zeros([K])
        for k in range(1,K + 1):
            gamma[k - 1] = log_real(theta[doc[n - 1] - 1, k - 1]) + log_real(
            phi[k - 1, w[n - 1] - 1])
        factor(f'expr__{n}__3', log_sum_exp_array(gamma))

