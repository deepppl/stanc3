import pyro.distributions as d
from torch.distributions import constraints, transform_to
from torch.distributions.constraints import Constraint
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


# class lower_constrained_improper_uniform(improper_uniform):
#     def __init__(self, lower_bound=0, shape=None):
#         self.lower_bound = lower_bound if shape == None else lower_bound * torch.ones(shape)
#         super(lower_constrained_improper_uniform, self).__init__(shape)
#         self.support = constraints.greater_than_eq(lower_bound)

#     def sample(self, *args, **kwargs):
#         s = d.Uniform(self.lower_bound, self.lower_bound + 2).sample(*args, **kwargs)
#         return s

class lower_constrained_improper_uniform(improper_uniform):
    def __init__(self, lower_bound=0, shape=None):
        super().__init__(shape)
        self._cstr = constraints.greater_than_eq(lower_bound)
        self.support = constraints.greater_than_eq(lower_bound)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform_to(self._cstr)(s)


# class upper_constrained_improper_uniform(improper_uniform):
#     def __init__(self, upper_bound=0.0, shape=None):
#         self.upper_bound = upper_bound if shape == None else upper_bound * torch.ones(shape)
#         super(upper_constrained_improper_uniform, self).__init__(shape)
#         self.support = constraints.less_than(upper_bound)

#     def sample(self, *args, **kwargs):
#         s = d.Uniform(self.upper_bound - 2.0, self.upper_bound).sample(*args, **kwargs)
#         return s

class _LessThanEq(Constraint):
    """
    Constrain to a real half line `[-inf, upper_bound]`.
    """
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
    def check(self, value):
        return value <= self.upper_bound
    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(upper_bound={})'.format(self.upper_bound)
        return fmt_string

class upper_constrained_improper_uniform(improper_uniform):
    def __init__(self, upper_bound=0.0, shape=None):
        super().__init__(shape)
        self._cstr = _LessThanEq(upper_bound)
        self.support = constraints.greater_than_eq(upper_bound)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform_to(self._cstr)(s)

class simplex_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=None):
        super().__init__(shape)
        self.support = constraints.simplex

    _transform = transform_to(constraints.simplex)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return self._transform(s)

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
        return self._transform(s)


class cholesky_factor_corr_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=None):
        super().__init__(shape[0])
        self.support = constraints.lower_cholesky

    _transform = CorrLCholeskyTransform()

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return self._transform(s)


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
def dirichlet_lpdf_vector_vector(y, x):
    return d.Dirichlet(x).log_prob(y)
multi_normal = d.MultivariateNormal
logistic = d.LogisticNormal
cauchy = d.Cauchy
lognormal = d.LogNormal
double_exponential = d.Laplace
exponential = d.Exponential
pareto = d.Pareto
neg_binomial_2 = d.GammaPoisson
def neg_binomial_2_lpmf(y, mu, phi):
    y = y.type(torch.float) if isinstance(y, torch.LongTensor) else array(y, dtype=torch.long)
    return d.GammaPoisson(mu, phi).log_prob(y)
neg_binomial_2_lpmf_int_real_real = neg_binomial_2_lpmf
neg_binomial_2_lpmf_array_vector_real = neg_binomial_2_lpmf
def neg_binomial_2_rng(mu, phi):
    return d.GammaPoisson(mu, phi).sample()
neg_binomial_2_rng_real_real = neg_binomial_2_rng
neg_binomial_2_rng_vector_real = neg_binomial_2_rng

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
