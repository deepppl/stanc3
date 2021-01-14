

import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N=None, Ngrps=None, grp_index=None, p=None):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['p'] = N
    ___shape['Ngrps'] = ()
    ___shape['grp_index'] = N
    ___shape['sigmaGrp'] = Ngrps
    ___shape['muGrp'] = Ngrps
    sigmaGrp = sample('sigmaGrp', dist.Uniform(0.0001, 100.0, Ngrps))
    muGrp = sample('muGrp', dist.Uniform(-100, 1000.0, Ngrps))
    ___shape['grpi'] = ()
    for i in range(1, N + 1):
        grpi = grp_index[i - 1]
        sample('p' + '__{}'.format(i - 1) + '__1', dist.LogisticNormal(
            muGrp[grpi - 1], sigmaGrp[grpi - 1]), obs=p[i - 1])
