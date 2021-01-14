
# Adapted from https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
from pyro import distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.autograd import Variable

from observations import text8
import random
import string

import torch.utils.data.dataloader as dataloader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

import deepppl
import os
import pytest

data_dir = os.path.join(os.environ.get("DATA_DIR", '.'), "data")
log_dir = os.path.join(os.environ.get("DATA_DIR", '.'), "log")

all_characters = string.ascii_lowercase + ' '
all_characters_dict = {z: i for (i, z) in enumerate(all_characters)}
n_characters = len(all_characters)

def char_tensor(string):
    return tensor([all_characters_dict[x] for x in string]).long()


def generator(input, seq_len=200):
    while True:
        start_index = random.randint(0, len(input) - seq_len)
        end_index = start_index + seq_len + 1
        chunk = input[start_index:end_index]
        inp = char_tensor(chunk[:-1])
        target = char_tensor(chunk[1:])
        yield inp, target

def build_rnn():
    # Model

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, n_layers=1):
            super(RNN, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers

            self.encoder = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
            self.decoder = nn.Linear(hidden_size, output_size)
            
        def forward(self, input):
            input = self.encoder(input)
            ## batch_size of 1, len(input) is the size of the series
            output, _ = self.gru(input.view(len(input), 1, -1))
            output = self.decoder(output.squeeze(dim=1))
            return nn.functional.log_softmax(output, dim=-1)

    hidden_size = 102
    return RNN(n_characters, hidden_size, n_characters, n_layers=1)




num_samples = 1


def evaluate_samples(svi, first_letter='w', predict_len=100, temperature=0.8):
    sampled_rnn = svi.posterior(num_samples)

    input = char_tensor(first_letter).view(len(first_letter), -1)
    predicted = first_letter

    for i in range(predict_len):
        outputs = [rnn(input)[-1,:] for rnn in sampled_rnn]

        # Sample from the network as a multinomial distribution
        probs = [o.div(temperature).exp() for o in outputs]
        mean_dist = torch.mean(torch.stack(probs), 0)
        top_i = torch.multinomial(mean_dist, 1)[0]

        predicted_char = all_characters[top_i]
        predicted += predicted_char
        input = char_tensor(predicted)

    return predicted

@pytest.mark.xfail(strict=False, reason="This currently fails with type inference.  Reasons not yet investigated.")
def test_lstm_inference():
    rnn = build_rnn()
    model = deepppl.PyroModel(model_file = 'deepppl/tests/good/lstm_modified.stan', rnn=rnn)
    adam_params = {
        "lr": .1, #0.001,
        "betas": (0.96, 0.999),
        "clip_norm": 20.0,
        "lrd": 0.99996,
        "weight_decay": 2.0}

    adam = ClippedAdam(adam_params)
    svi = model.svi(optimizer=adam, loss=Trace_ELBO(num_particles=5))
    data_, _, _ = text8(data_dir)
    data = generator(data_)
    print_every = 100
    epoch = 0
    for i, (input, target) in enumerate(data):
            loss = svi.step(target, input, n_characters)
            loss = loss / len(input) 

            if (i+1) % print_every == 0:
                print('[epoch {}, iteration {}, loss = {}]'.format(epoch, i, loss))
                print(evaluate_samples(svi), '\n')
                break