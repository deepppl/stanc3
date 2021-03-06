from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, register_network, random_module

def convert_inputs(inputs):
    nz = inputs['nz']
    x = array(inputs['x'], dtype=dtype_long)
    decoder = inputs['decoder']
    encoder = inputs['encoder']
    return { 'nz': nz, 'x': x, 'encoder': encoder, 'decoder': decoder }

def model(*, nz, x, decoder, encoder):
    # Networks
    register_network('decoder', decoder)
    # Parameters
    z = sample('z', improper_uniform(shape=[nz]))
    # Model
    observe('_z__1', normal(0, 1), z)
    mu = decoder(z)
    observe('_x__2', bernoulli(mu), x)

def guide(*, nz, x, decoder, encoder):
    # Networks
    register_network('encoder', encoder)
    # Guide
    encoded = encoder(x)
    mu_z = encoded[1 - 1]
    sigma_z = encoded[2 - 1]
    z = sample('z', normal(mu_z, sigma_z))
    return { 'z': z }
