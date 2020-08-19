import pyro.distributions as d
from torch.distributions import constraints
import torch

class improper_uniform(d.Normal):
    def __init__(self, shape=None):
        zeros = torch.zeros(shape) if shape else 0
        ones = torch.ones(shape) if shape else 1
        super(improper_uniform, self).__init__(zeros, ones)

    def log_prob(self, x):
        return x.new_zeros(x.shape)


class lower_constrained_improper_uniform(improper_uniform):
    def __init__(self, lower_bound=0, shape=None):
        self.lower_bound = lower_bound
        super(lower_constrained_improper_uniform, self).__init__(shape)
        self.support = constraints.greater_than(lower_bound)

    def sample(self, *args, **kwargs):
        s = d.Uniform(self.lower_bound, self.lower_bound + 2).sample(*args, **kwargs)
        return s


class upper_constrained_improper_uniform(improper_uniform):
    def __init__(self, upper_bound=0.0, shape=None):
        self.upper_bound = upper_bound
        super(upper_constrained_improper_uniform, self).__init__(shape)
        self.support = constraints.less_than(upper_bound)

    def sample(self, *args, **kwargs):
        s = d.Uniform(self.upper_bound - 2.0, self.upper_bound).sample(*args, **kwargs)
        return s

uniform = d.Uniform
beta = d.Beta
bernoulli = d.Bernoulli
normal = d.Normal
student_t = d.StudentT
inv_gamma = d.InverseGamma
gamma = d.Gamma

def categorical_logits(logits):
    return d.Categorical(logits=logits)

def bernoulli_logit(logits):
    return d.Bernoulli(logits=logits)

def binomial_logit(n, logits):
    return d.Binomial(n, logits=logits)

def poisson_log(alpha):
    return d.Poisson(torch.exp(alpha))
