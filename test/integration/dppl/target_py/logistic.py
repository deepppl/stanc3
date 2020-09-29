from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network
from runtimes.pyro.stanlib import inv_logit_real

def convert_inputs(inputs):
    N = inputs['N']
    M = inputs['M']
    y = array(inputs['y'], dtype=dtype_long)
    x = array(inputs['x'], dtype=dtype_float)
    return { 'N': N, 'M': M, 'y': y, 'x': x }

def model(*, N, M, y, x):
    # Parameters
    beta = sample('beta', improper_uniform(shape=[M]))
    # Model
    for m in range(1,M + 1):
        observe(f'beta__{m}__1', cauchy(array(0.0, dtype=dtype_float),
                                        array(2.5, dtype=dtype_float)), beta[
        m - 1])
    for n in range(1,N + 1):
        observe(f'y__{n}__2', bernoulli(inv_logit_real(matmul(x[n - 1], beta))), y[
        n - 1])

