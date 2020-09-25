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
def normal_lpdf(y, mu, sigma):
    return d.Normal(mu, sigma).log_prob(y)
normal_lpdf_real_real_real = normal_lpdf
normal_lpdf_real_int_int = normal_lpdf
normal_lpdf_vector_int_int = normal_lpdf
normal_lpdf_vector_real_real = normal_lpdf
normal_lpdf_vector_vector_real = normal_lpdf
student_t = d.StudentT
inv_gamma = d.InverseGamma
gamma = d.Gamma
dirichlet = d.Dirichlet
multi_normal = d.MultivariateNormal
logistic = d.LogisticNormal
cauchy = d.Cauchy
lognormal = d.LogNormal
double_exponential = d.Laplace

def categorical_logit(logits):
    return d.Categorical(logits=logits)

def bernoulli_logit(logits):
    return d.Bernoulli(logits=logits)

def binomial_logit(n, logits):
    return d.Binomial(n, logits=logits)

def poisson_log(alpha):
    return d.Poisson(torch.exp(alpha))

def std_normal():
    return d.Normal(0,1)
