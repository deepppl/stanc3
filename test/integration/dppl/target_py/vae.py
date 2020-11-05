from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network, random_module

def convert_inputs(inputs):
    nz = inputs['nz']
    x = array(inputs['x'], dtype=dtype_long)
    return { 'nz': nz, 'x': x }

def model(*, nz, x, decoder, encoder):
    # Networks
    register_network('decoder', decoder)
    # Parameters
    z = sample('z', improper_uniform(shape=[nz]))
    # Model
    observe('z__1', normal(0, 1), z)
    mu = decoder(z)
    for i in range(1,28 + 1):
        observe(f'x__{i}__2', bernoulli(mu[i - 1]), x[i - 1])

def guide(*, nz, x, decoder, encoder):
    # Networks
    register_network('encoder', encoder)
    # Guide
    encoded = encoder(x)
    mu_z = encoded[1 - 1]
    sigma_z = encoded[2 - 1]
    z = sample('z', normal(mu_z, sigma_z))
    return { z }
