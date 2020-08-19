
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_mlp(batch_size=None, imgs=None, labels=None):
    guide_mlp = {}
    l1wloc = pyro.param('l1wloc', randn(mlp.l1.weight.size()))
    l1wscale = pyro.param('l1wscale', randn(mlp.l1.weight.size()))
    guide_mlp['l1.weight'] = dist.Normal(l1wloc, softplus(l1wscale))
    l1bloc = pyro.param('l1bloc', randn(mlp.l1.bias.size()))
    l1bscale = pyro.param('l1bscale', randn(mlp.l1.bias.size()))
    guide_mlp['l1.bias'] = dist.Normal(l1bloc, softplus(l1bscale))
    l2wloc = pyro.param('l2wloc', randn(mlp.l2.weight.size()))
    l2wscale = pyro.param('l2wscale', randn(mlp.l2.weight.size()))
    guide_mlp['l2.weight'] = dist.Normal(l2wloc, softplus(l2wscale))
    l2bloc = pyro.param('l2bloc', randn(mlp.l2.bias.size()))
    l2bscale = pyro.param('l2bscale', randn(mlp.l2.bias.size()))
    guide_mlp['l2.bias'] = dist.Normal(l2bloc, softplus(l2bscale))
    lifted_mlp = pyro.random_module('mlp', mlp, guide_mlp)
    return lifted_mlp()


def prior_mlp(batch_size=None, imgs=None, labels=None):
    prior_mlp = {}
    prior_mlp['l1.weight'] = ImproperUniform(shape=mlp.l1.weight.shape)
    prior_mlp['l1.bias'] = ImproperUniform(shape=mlp.l1.bias.shape)
    prior_mlp['l2.weight'] = ImproperUniform(shape=mlp.l2.weight.shape)
    prior_mlp['l2.bias'] = ImproperUniform(shape=mlp.l2.bias.shape)
    lifted_mlp = pyro.random_module('mlp', mlp, prior_mlp)
    return lifted_mlp()


def model(batch_size=None, imgs=None, labels=None):
    mlp = prior_mlp(batch_size, imgs, labels)
    model_mlp = dict(mlp.named_parameters())
    logits = zeros(batch_size)
    sample('model_mlp' + '__{}'.format('l1.weight') + '__1', dist.
        Normal(zeros(mlp.l1.weight.shape), ones(mlp.l1.weight.shape)), obs=
        model_mlp['l1.weight'])
    sample('model_mlp' + '__{}'.format('l1.bias') + '__2', dist.Normal
        (zeros(mlp.l1.bias.shape), ones(mlp.l1.bias.shape)), obs=model_mlp[
        'l1.bias'])
    sample('model_mlp' + '__{}'.format('l2.weight') + '__3', dist.
        Normal(zeros(mlp.l2.weight.shape), ones(mlp.l2.weight.shape)), obs=
        model_mlp['l2.weight'])
    sample('model_mlp' + '__{}'.format('l2.bias') + '__4', dist.Normal
        (zeros(mlp.l2.bias.shape), ones(mlp.l2.bias.shape)), obs=model_mlp[
        'l2.bias'])
    logits = mlp(imgs)
    sample('labels' + '__5', categorical_logits(logits), obs=labels)
