from runtimes.pyro.distributions import *
from runtimes.pyro.dppllib import sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, register_network, random_module
from runtimes.pyro.stanlib import log1p_exp_array

def convert_inputs(inputs):
    nx = inputs['nx']
    nh = inputs['nh']
    ny = inputs['ny']
    batch_size = inputs['batch_size']
    imgs = array(inputs['imgs'], dtype=dtype_long)
    labels = array(inputs['labels'], dtype=dtype_long)
    return { 'nx': nx, 'nh': nh, 'ny': ny, 'batch_size': batch_size,
             'imgs': imgs, 'labels': labels }

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
    observe('mlp.l1.weight__1', normal(0, 1), mlp_['l1.weight'])
    observe('mlp.l1.bias__2', normal(0, 1), mlp_['l1.bias'])
    observe('mlp.l2.weight__3', normal(0, 1), mlp_['l2.weight'])
    observe('mlp.l2.bias__4', normal(0, 1), mlp_['l2.bias'])
    logits = mlp(imgs)
    observe('labels__5', categorical_logit(logits), labels - 1)

def guide(*, nx, nh, ny, batch_size, imgs, labels, mlp):
    # Guide Parameters
    l1wloc = param('l1wloc', improper_uniform(shape=[nh, nx]).sample())
    l1wscale = param('l1wscale', improper_uniform(shape=[nh, nx]).sample())
    l1bloc = param('l1bloc', improper_uniform(shape=[nh]).sample())
    l1bscale = param('l1bscale', improper_uniform(shape=[nh]).sample())
    l2wloc = param('l2wloc', improper_uniform(shape=[ny, nh]).sample())
    l2wscale = param('l2wscale', improper_uniform(shape=[ny, nh]).sample())
    l2bloc = param('l2bloc', improper_uniform(shape=[ny]).sample())
    l2bscale = param('l2bscale', improper_uniform(shape=[ny]).sample())
    mlp_ = {}
    # Guide
    mlp_['l1.weight'] = normal(l1wloc, log1p_exp_array(l1wscale))
    mlp_['l1.bias'] = normal(l1bloc, log1p_exp_array(l1bscale))
    mlp_['l2.weight'] = normal(l2wloc, log1p_exp_array(l2wscale))
    mlp_['l2.bias'] = normal(l2bloc, log1p_exp_array(l2bscale))
    return { 'mlp': random_module('mlp', mlp, mlp_)(),  }
