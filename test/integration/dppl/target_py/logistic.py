from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float
from runtimes.pyro.stanlib import inv_logit_real

def convert_inputs(inputs):
    N = inputs['N']
    y = array(inputs['y'], dtype=dtype_long)
    M = inputs['M']
    x = array(inputs['x'], dtype=dtype_float)
    return { 'N': N, 'y': y, 'M': M, 'x': x }

def model(*, N, y, M, x):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[M]))
    # Model
    for m in range(1,M + 1):
        observe(f'beta__{m}__1', cauchy(array(0.0, dtype=dtype_float),
                                        array(2.5, dtype=dtype_float)), beta__[
        m - 1])
    for n in range(1,N + 1):
        observe(f'y__{n}__2', bernoulli(inv_logit_real(matmul(x[n - 1], beta__))), y[
        n - 1])

