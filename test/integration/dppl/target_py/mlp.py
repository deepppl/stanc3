from stanpyro.distributions import *
from stanpyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, register_network, random_module
from stanpyro.stanlib import exp_array

def convert_inputs(inputs):
    nx = inputs['nx']
    nh = inputs['nh']
    ny = inputs['ny']
    batch_size = inputs['batch_size']
    imgs = array(inputs['imgs'], dtype=dtype_long)
    labels = array(inputs['labels'], dtype=dtype_long)
    mlp = inputs['mlp']
    return { 'nx': nx, 'nh': nh, 'ny': ny, 'batch_size': batch_size,
             'imgs': imgs, 'labels': labels, 'mlp': mlp }

def prior_mlp(*, nx, nh, ny, batch_size, imgs, labels, mlp):
    mlp_ = {}
    mlp_['l1.weight'] = improper_uniform(shape=[nh, nx])
    mlp_['l1.bias'] = improper_uniform(shape=[nh])
    mlp_['l2.weight'] = improper_uniform(shape=[ny, nh])
    mlp_['l2.bias'] = improper_uniform(shape=[ny])
    return random_module('mlp', mlp, mlp_)()

def model(*, nx, nh, ny, batch_size, imgs, labels, mlp):
    # Parameters
    mlp = prior_mlp(nx=nx, nh=nh, ny=ny, batch_size=batch_size, imgs=imgs,
                    labels=labels, mlp=mlp)
    mlp_ = dict(mlp.named_parameters())
    # Model
    observe('_mlp.l1.weight__1', normal(0, 1), mlp_['l1.weight'])
    observe('_mlp.l1.bias__2', normal(0, 1), mlp_['l1.bias'])
    observe('_mlp.l2.weight__3', normal(0, 1), mlp_['l2.weight'])
    observe('_mlp.l2.bias__4', normal(0, 1), mlp_['l2.bias'])
    lambda__ = mlp(imgs)
    observe('_labels__5', categorical_logit(lambda__), labels - 1)

def guide(*, nx, nh, ny, batch_size, imgs, labels, mlp):
    # Guide Parameters
    w1_mu = param('w1_mu', improper_uniform(shape=[nh, nx]).sample())
    w1_sgma = param('w1_sgma', improper_uniform(shape=[nh, nx]).sample())
    b1_mu = param('b1_mu', improper_uniform(shape=[nh]).sample())
    b1_sgma = param('b1_sgma', improper_uniform(shape=[nh]).sample())
    w2_mu = param('w2_mu', improper_uniform(shape=[ny, nh]).sample())
    w2_sgma = param('w2_sgma', improper_uniform(shape=[ny, nh]).sample())
    b2_mu = param('b2_mu', improper_uniform(shape=[ny]).sample())
    b2_sgma = param('b2_sgma', improper_uniform(shape=[ny]).sample())
    mlp_ = {}
    # Guide
    mlp_['l1.weight'] = normal(w1_mu, exp_array(w1_sgma))
    mlp_['l1.bias'] = normal(b1_mu, exp_array(b1_sgma))
    mlp_['l2.weight'] = normal(w2_mu, exp_array(w2_sgma))
    mlp_['l2.bias'] = normal(b2_mu, exp_array(b2_sgma))
    return { 'mlp': random_module('mlp', mlp, mlp_)(),  }
