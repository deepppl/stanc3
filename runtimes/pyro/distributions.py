import pyro.distributions as d
from torch.distributions import constraints, transform_to
from pyro.distributions.transforms import CorrLCholeskyTransform
from torch import sort
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
        self.support = constraints.greater_than_eq(lower_bound)

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

class simplex_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=None):
        super().__init__(shape)
        self.support = constraints.simplex

    _transform = transform_to(constraints.simplex)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return _transform(s)

class unit_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=None):
        super().__init__(shape)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return s / torch.norm(s)

class ordered_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=None):
        super().__init__(shape)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return sort(s)

class positive_ordered_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=None):
        super().__init__(shape)
        self.support = constraints.positive

    _to_positive = transform_to(constraints.positive)
    def _transform(x):
        sort(_to_positive(x))

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return _transform(s)


class cholesky_factor_corr_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=None):
        super().__init__(shape[0])
        self.support = constraints.lower_cholesky

    _transform = CorrLCholeskyTransform()

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return _transform(s)


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
def student_t_lpdf(y, nu, mu, sigma):
    return d.StudentT(nu, mu, sigma).log_prob(y)
def student_t_lccdf(y, nu, mu, sigma):
    return torch.log(d.StudentT(nu, mu, sigma).icdf(y))
student_t_lpdf_real_int_int_int = student_t_lpdf
student_t_lccdf_int_int_int_int = student_t_lccdf
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
