

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    if 1.0 == 0.0:
        a: 'real' = 1.0
