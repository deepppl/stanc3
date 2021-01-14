

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_mlp(img=None, label=None):
    guide_mlp = {}
    w1_mu = pyro.param('w1_mu', (2 - -2) * rand(()) + -2)
    w1_sgma = pyro.param('w1_sgma', (2 - -2) * rand(()) + -2)
    b1_mu = pyro.param('b1_mu', (2 - -2) * rand(()) + -2)
    b1_sgma = pyro.param('b1_sgma', (2 - -2) * rand(()) + -2)
    w2_mu = pyro.param('w2_mu', (2 - -2) * rand(()) + -2)
    w2_sgma = pyro.param('w2_sgma', (2 - -2) * rand(()) + -2)
    b2_mu = pyro.param('b2_mu', (2 - -2) * rand(()) + -2)
    b2_sgma = pyro.param('b2_sgma', (2 - -2) * rand(()) + -2)
    guide_mlp['l1.weight'] = dist.Normal(w1_mu, exp(w1_sgma))
    guide_mlp['l1.bias'] = dist.Normal(b1_mu, exp(b1_sgma))
    guide_mlp['l2.weight'] = dist.Normal(w2_mu, exp(w2_sgma))
    guide_mlp['l2.bias'] = dist.Normal(b2_mu, exp(b2_sgma))
    lifted_mlp = pyro.random_module('mlp', mlp, guide_mlp)
    return lifted_mlp()


def prior_mlp(img=None, label=None):
    prior_mlp = {}
    prior_mlp['l1.weight'] = ImproperUniform(shape=mlp.l1.weight.shape)
    prior_mlp['l1.bias'] = ImproperUniform(shape=mlp.l1.bias.shape)
    prior_mlp['l2.weight'] = ImproperUniform(shape=mlp.l2.weight.shape)
    prior_mlp['l2.bias'] = ImproperUniform(shape=mlp.l2.bias.shape)
    lifted_mlp = pyro.random_module('mlp', mlp, prior_mlp)
    return lifted_mlp()


def model(img=None, label=None):
    mlp = prior_mlp(img, label)
    model_mlp = dict(mlp.named_parameters())
    logits = zeros(10)
    sample('model_mlp' + '__{}'.format('l1.weight') + '__1', dist.
        Normal(0, 1), obs=model_mlp['l1.weight'])
    sample('model_mlp' + '__{}'.format('l1.bias') + '__2', dist.Normal
        (0, 1), obs=model_mlp['l1.bias'])
    sample('model_mlp' + '__{}'.format('l2.weight') + '__3', dist.
        Normal(0, 1), obs=model_mlp['l2.weight'])
    sample('model_mlp' + '__{}'.format('l2.bias') + '__4', dist.Normal
        (0, 1), obs=model_mlp['l2.bias'])
    logits = mlp(img)
    sample('label' + '__5', categorical_logits(logits), obs=label)
