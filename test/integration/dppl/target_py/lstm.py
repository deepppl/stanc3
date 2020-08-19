
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_rnn(category=None, input=None, n_characters=None):
    ___shape = {}
    ___shape['n_characters'] = ()
    ___shape['input'] = n_characters
    ___shape['category'] = n_characters
    ___shape['ewl'] = rnn.encoder.weight.shape
    ___shape['ews'] = rnn.encoder.weight.shape
    ___shape['gw1l'] = rnn.gru.weight_ih_l0.shape
    ___shape['gw1s'] = rnn.gru.weight_ih_l0.shape
    ___shape['gw2l'] = rnn.gru.weight_hh_l0.shape
    ___shape['gw2s'] = rnn.gru.weight_hh_l0.shape
    ___shape['gb1l'] = rnn.gru.bias_ih_l0.shape
    ___shape['gb1s'] = rnn.gru.bias_ih_l0.shape
    ___shape['gb2l'] = rnn.gru.bias_hh_l0.shape
    ___shape['gb2s'] = rnn.gru.bias_hh_l0.shape
    ___shape['dwl'] = rnn.decoder.weight.shape
    ___shape['dws'] = rnn.decoder.weight.shape
    ___shape['dbl'] = rnn.decoder.bias.shape
    ___shape['dbs'] = rnn.decoder.bias.shape
    guide_rnn = {}
    ewl = pyro.param('ewl', 0.001 * randn(___shape['ewl']))
    ews = pyro.param('ews', randn(___shape['ews']) - 10.0)
    guide_rnn['encoder.weight'] = dist.Normal(ewl, exp(ews))
    gw1l = pyro.param('gw1l', 0.001 * randn(___shape['gw1l']))
    gw1s = pyro.param('gw1s', randn(___shape['gw1s']) - 10.0)
    guide_rnn['gru.weight_ih_l0'] = dist.Normal(gw1l, exp(gw1s))
    gw2l = pyro.param('gw2l', 0.001 * randn(___shape['gw2l']))
    gw2s = pyro.param('gw2s', randn(___shape['gw2s']) - 10.0)
    guide_rnn['gru.weight_hh_l0'] = dist.Normal(gw2l, exp(gw2s))
    gb1l = pyro.param('gb1l', 0.001 * randn(___shape['gb1l']))
    gb1s = pyro.param('gb1s', randn(___shape['gb1s']) - 10.0)
    guide_rnn['gru.bias_ih_l0'] = dist.Normal(gb1l, exp(gb1s))
    gb2l = pyro.param('gb2l', 0.001 * randn(___shape['gb2l']))
    gb2s = pyro.param('gb2s', randn(___shape['gb2s']) - 10.0)
    guide_rnn['gru.bias_hh_l0'] = dist.Normal(gb2l, exp(gb2s))
    dwl = pyro.param('dwl', 0.001 * randn(___shape['dwl']))
    dws = pyro.param('dws', randn(___shape['dws']) - 10.0)
    guide_rnn['decoder.weight'] = dist.Normal(dwl, exp(dws))
    dbl = pyro.param('dbl', 0.001 * randn(___shape['dbl']))
    dbs = pyro.param('dbs', randn(___shape['dbs']) - 10.0)
    guide_rnn['decoder.bias'] = dist.Normal(dbl, exp(dbs))
    lifted_rnn = pyro.random_module('rnn', rnn, guide_rnn)
    return lifted_rnn()


def prior_rnn(category=None, input=None, n_characters=None):
    ___shape = {}
    ___shape['n_characters'] = ()
    ___shape['input'] = n_characters
    ___shape['category'] = n_characters
    prior_rnn = {}
    prior_rnn['encoder.weight'] = ImproperUniform(rnn.encoder.weight.shape)
    prior_rnn['gru.weight_ih_l0'] = ImproperUniform(rnn.gru.weight_ih_l0.shape)
    prior_rnn['gru.weight_hh_l0'] = ImproperUniform(rnn.gru.weight_hh_l0.shape)
    prior_rnn['gru.bias_ih_l0'] = ImproperUniform(rnn.gru.bias_ih_l0.shape)
    prior_rnn['gru.bias_hh_l0'] = ImproperUniform(rnn.gru.bias_hh_l0.shape)
    prior_rnn['decoder.weight'] = ImproperUniform(rnn.decoder.weight.shape)
    prior_rnn['decoder.bias'] = ImproperUniform(rnn.decoder.bias.shape)
    lifted_rnn = pyro.random_module('rnn', rnn, prior_rnn)
    return lifted_rnn()


def model(category=None, input=None, n_characters=None):
    ___shape = {}
    ___shape['n_characters'] = ()
    ___shape['input'] = n_characters
    ___shape['category'] = n_characters
    rnn = prior_rnn(category, input, n_characters)
    model_rnn = dict(rnn.named_parameters())
    ___shape['logits'] = n_characters
    logits = zeros(___shape['logits'])
    sample('model_rnn' + '__{}'.format('encoder.weight') + '__1', dist.
        Normal(zeros(rnn.encoder.weight.shape), ones(rnn.encoder.weight.
        shape)), obs=model_rnn['encoder.weight'])
    sample('model_rnn' + '__{}'.format('gru.weight_ih_l0') + '__2', dist.
        Normal(zeros(rnn.gru.weight_ih_l0.shape), ones(rnn.gru.weight_ih_l0
        .shape)), obs=model_rnn['gru.weight_ih_l0'])
    sample('model_rnn' + '__{}'.format('gru.weight_hh_l0') + '__3', dist.
        Normal(zeros(rnn.gru.weight_hh_l0.shape), ones(rnn.gru.weight_hh_l0
        .shape)), obs=model_rnn['gru.weight_hh_l0'])
    sample('model_rnn' + '__{}'.format('gru.bias_ih_l0') + '__4', dist.
        Normal(zeros(rnn.gru.bias_ih_l0.shape), ones(rnn.gru.bias_ih_l0.
        shape)), obs=model_rnn['gru.bias_ih_l0'])
    sample('model_rnn' + '__{}'.format('gru.bias_hh_l0') + '__5', dist.
        Normal(zeros(rnn.gru.bias_hh_l0.shape), ones(rnn.gru.bias_hh_l0.
        shape)), obs=model_rnn['gru.bias_hh_l0'])
    sample('model_rnn' + '__{}'.format('decoder.weight') + '__6', dist.
        Normal(zeros(rnn.decoder.weight.shape), ones(rnn.decoder.weight.
        shape)), obs=model_rnn['decoder.weight'])
    sample('model_rnn' + '__{}'.format('decoder.bias') + '__7', dist.
        Normal(zeros(rnn.decoder.bias.shape), ones(rnn.decoder.bias.shape)),
        obs=model_rnn['decoder.bias'])
    logits = rnn(input)
    sample('category' + '__8', categorical_logits(logits), obs=category)
