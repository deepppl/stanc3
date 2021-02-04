from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap

def convert_inputs(inputs):
    N_mis = inputs['N_mis']
    N_obs = inputs['N_obs']
    y_obs = array(inputs['y_obs'], dtype=dtype_float)
    return { 'N_mis': N_mis, 'N_obs': N_obs, 'y_obs': y_obs }

def model(*, N_mis, N_obs, y_obs):
    # Parameters
    mu = sample('mu', improper_uniform(shape=[]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    y_mis = sample('y_mis', improper_uniform(shape=[N_mis]))
    # Model
    observe('_y_obs__1', normal(mu, sigma), y_obs)
    observe('_y_mis__2', normal(mu, sigma), y_mis)

