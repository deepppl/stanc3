import pyro.distributions as d
from torch.distributions import constraints, transform_to
from torch.distributions.constraints import Constraint
from pyro.distributions.transforms import CorrLCholeskyTransform
from torch import sort
from torch import log as tlog
from torch import exp as texp
import torch

## Utility functions
def _cast1(f):
    def f_casted(y, *args):
        y = y.type(torch.float) if isinstance(y, torch.LongTensor) else array(y, dtype=torch.float)
        return f(y, *args)
    return f_casted

def _lpdf(d):
    def lpdf(y, *args):
        return d(*args).log_prob(y)
    return lpdf

def _lpmf(d):
    def lpmf(y, *args):
        return d(*args).log_prob(y)
    return lpmf

def _cdf(d):
    def lccdf(y, *args):
        return d(*args).cdf(y)
    return lccdf

def _lcdf(d):
    def lccdf(y, *args):
        return tlog(d(*args).cdf(y))
    return lccdf

def _lccdf(d):
    # XXX TODO: check id correct XXX
    def lccdf(y, *args):
        return tlog(d(*args).icdf(y))
    return lccdf

def _rng(d):
    def rng(*args):
        return d(*args).sample()
    return rng


## Priors

class improper_uniform(d.Normal):
    def __init__(self, shape=None):
        zeros = torch.zeros(shape) if shape else 0
        ones = torch.ones(shape) if shape else 1
        super(improper_uniform, self).__init__(zeros, ones)

    def log_prob(self, x):
        return x.new_zeros(x.shape)

class lower_constrained_improper_uniform(improper_uniform):
    def __init__(self, lower_bound=0, shape=None):
        super().__init__(shape)
        self._cstr = constraints.greater_than_eq(lower_bound)
        self.support = constraints.greater_than_eq(lower_bound)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform_to(self._cstr)(s)

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
        s, _indices = sort(s)
        return s

class positive_ordered_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=None):
        super().__init__(shape)
        self.support = constraints.positive

    _to_positive = transform_to(constraints.positive)
    def _transform(self, x):
        return sort(self._to_positive(x)).values

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


## Stan distributions

## 19 Continuous Distributions on [0, 1]

## 19.1 Beta Distribution

# real beta_lpdf(reals theta | reals alpha, reals beta)
# The log of the beta density of theta in [0,1] given positive prior
# successes (plus one) alpha and prior failures (plus one) beta

beta = d.Beta
beta_lpdf = _lpdf(beta)
beta_cdf = _cdf(beta)
beta_lcdf = _lcdf(beta)
beta_lccdf = _lccdf(beta)
beta_rng = _rng(beta)


## 12 Binary Distributions

## 12.1 Bernoulli Distribution

# real bernoulli_lpmf(ints y | reals theta)
# The log Bernoulli probability mass of y given chance of success theta

bernoulli = d.Bernoulli
bernoulli_lpmf = _lpmf(bernoulli)
bernoulli_cdf = _cdf(bernoulli)
bernoulli_lcdf = _lcdf(bernoulli)
bernoulli_lccdf = _lccdf(bernoulli)
bernoulli_rng = _rng(bernoulli)

## 12.2 Bernoulli Distribution, Logit Parameterization

# real bernoulli_logit_lpmf(ints y | reals alpha)
# The log Bernoulli probability mass of y given chance of success inv_logit(alpha)

bernoulli_logit = lambda logits: d.Bernoulli(logits=logits)
bernoulli_logit_lpmf = _lpmf(bernoulli_logit)


## 13 Bounded Discrete Distributions

## 13.2 Binomial Distribution, Logit Parameterization

# real binomial_logit_lpmf(ints n | ints N, reals alpha)
# The log binomial probability mass of n successes in N trials given logit-scaled chance of success alpha

binomial_logit = lambda n, logits: d.Binomial(n, logits=logits)
binomial_logit_lpmf = _lpmf(binomial_logit)

## 13.5 Categorical Distribution

# real categorical_lpmf(ints y | vector theta)
# The log categorical probability mass function with outcome(s) y in 1:N given N-vector of outcome probabilities theta. The parameter theta must have non-negative entries that sum to one, but it need not be a variable declared as a simplex.

categorical = d.Categorical
categorical_lpmf = _lpmf(categorical)
categorical_rng = _rng(categorical)

# real categorical_logit_lpmf(ints y | vector beta)
# The log categorical probability mass function with outcome(s) y in 1:N
# given log-odds of outcomes beta.

categorical_logit = lambda logits: d.Categorical(logits=logits)
categorical_logit_lpmf = _lpmf(categorical_logit)
categorical_logit_rng = _rng(categorical_logit)

## 14 Unbounded Discrete Distributions

## 14.2 Negative Binomial Distribution (alternative parameterization)

# real neg_binomial_2_lpmf(ints n | reals mu, reals phi)
# The negative binomial probability mass of n given location mu and precision phi.

neg_binomial_2 = d.GammaPoisson
neg_binomial_2_lpmf = _lpmf(_cast1(neg_binomial_2))
neg_binomial_2_cdf = _cdf(_cast1(neg_binomial_2))
neg_binomial_2_lcdf = _lcdf(_cast1(neg_binomial_2))
neg_binomial_2_lccdf = _lccdf(_cast1(neg_binomial_2))
neg_binomial_2_rng = _rng(neg_binomial_2)

## 14.6 Poisson Distribution, Log Parameterization

# real poisson_log_lpmf(ints n | reals alpha)
# The log Poisson probability mass of n given log rate alpha

poisson = d.Poisson
poisson_log = lambda alpha: d.Poisson(texp(alpha))
poisson_log_lpmf = _lpmf(poisson_log)
poisson_log_rng = _rng(poisson_log)


## 16 Unbounded Continuous Distributions

# 16.1 Normal Distribution

# real normal_lpdf(reals y | reals mu, reals sigma)
# The log of the normal density of y given location mu and scale sigma

normal = d.Normal
normal_lpdf = _lpdf(normal)
normal_cdf = _cdf(normal)
normal_lcdf = _lcdf(normal)
normal_lccdf = _lccdf(normal)
normal_rng = _rng(normal)

# real std_normal_lpdf(reals y)
# The standard normal (location zero, scale one) log probability density of y.

std_normal = lambda : d.Normal(0,1)
std_normal_lpdf = _lpdf(std_normal)
std_normal_cdf = _cdf(std_normal)
std_normal_lcdf = _lcdf(std_normal)
std_normal_lccdf = _lccdf(std_normal)
std_normal_rng = _rng(std_normal)

## 16.5 Student-T Distribution

# real student_t_lpdf(reals y | reals nu, reals mu, reals sigma)
# The log of the Student-t density of y given degrees of freedom nu, location mu, and scale sigma

student_t = d.StudentT
student_t_lpdf = _lpdf(student_t)
student_t_cdf = _cdf(student_t)
student_t_lcdf = _lcdf(student_t)
student_t_lccdf = _lccdf(student_t)
student_t_rng = _rng(student_t)

## 16.6 Cauchy Distribution

# real cauchy_lpdf(reals y | reals mu, reals sigma)
# The log of the Cauchy density of y given location mu and scale sigma

cauchy = d.Cauchy
cauchy_lpdf = _lpdf(cauchy)
cauchy_cdf = _cdf(cauchy)
cauchy_lcdf = _lcdf(cauchy)
cauchy_lccdf = _lccdf(cauchy)
cauchy_rng = _rng(cauchy)

## 16.7 Double Exponential (Laplace) Distribution

# real double_exponential_lpdf(reals y | reals mu, reals sigma)
# The log of the double exponential density of y given location mu and scale sigma

double_exponential = d.Laplace
double_exponential_lpdf = _lpdf(double_exponential)
double_exponential_cdf = _cdf(double_exponential)
double_exponential_lcdf = _lcdf(double_exponential)
double_exponential_lccdf = _lccdf(double_exponential)
double_exponential_rng = _rng(double_exponential)

## 16.8 Logistic Distribution

# real logistic_lpdf(reals y | reals mu, reals sigma)
# The log of the logistic density of y given location mu and scale sigma

logistic = d.LogisticNormal
logistic_lpdf = _lpdf(logistic)
logistic_cdf = _cdf(logistic)
logistic_lcdf = _lcdf(logistic)
logistic_lccdf = _lccdf(logistic)
logistic_rng = _rng(logistic)


## 17 Positive Continuous Distributions

## 17.1 Lognormal Distribution

# real lognormal_lpdf(reals y | reals mu, reals sigma)
# The log of the lognormal density of y given location mu and scale sigma

lognormal = d.LogNormal
lognormal_lpdf = _lpdf(lognormal)
lognormal_cdf = _cdf(lognormal)
lognormal_lcdf = _lcdf(lognormal)
lognormal_lccdf = _lccdf(lognormal)
lognormal_rng = _rng(lognormal)

## 17.5 Exponential Distribution

# real exponential_lpdf(reals y | reals beta)
# The log of the exponential density of y given inverse scale beta

exponential = d.Exponential
exponential_lpdf = _lpdf(exponential)
exponential_cdf = _cdf(exponential)
exponential_lcdf = _lcdf(exponential)
exponential_lccdf = _lccdf(exponential)
exponential_rng = _rng(exponential)

## 17.6 Gamma Distribution

# real gamma_lpdf(reals y | reals alpha, reals beta)
# The log of the gamma density of y given shape alpha and inverse scale beta

gamma = d.Gamma
gamma_lpdf = _lpdf(gamma)
gamma_cdf = _cdf(gamma)
gamma_lcdf = _lcdf(gamma)
gamma_lccdf = _lccdf(gamma)
gamma_rng = _rng(gamma)

## 17.7 Inverse Gamma Distribution

# real inv_gamma_lpdf(reals y | reals alpha, reals beta)
# The log of the inverse gamma density of y given shape alpha and scale beta

inv_gamma = d.InverseGamma
inv_gamma_lpdf = _lpdf(inv_gamma)
inv_gamma_cdf = _cdf(inv_gamma)
inv_gamma_lcdf = _lcdf(inv_gamma)
inv_gamma_lccdf = _lccdf(inv_gamma)
inv_gamma_rng = _rng(inv_gamma)

## 18 Positive Lower-Bounded Distributions

## 18.1 Pareto Distribution

# real pareto_lpdf(reals y | reals y_min, reals alpha)
# The log of the Pareto density of y given positive minimum value y_min and shape alpha

pareto = d.Pareto
pareto_lpdf = _lpdf(pareto)
pareto_cdf = _cdf(pareto)
pareto_lcdf = _lcdf(pareto)
pareto_lccdf = _lccdf(pareto)
pareto_rng = _rng(pareto)


## 21 Bounded Continuous Probabilities

## 21.1 Uniform Distribution

# real uniform_lpdf(reals y | reals alpha, reals beta)
# The log of the uniform density of y given lower bound alpha and upper bound beta

uniform = d.Uniform
uniform_lpdf = _lpdf(uniform)
uniform_cdf = _cdf(uniform)
uniform_lcdf = _lcdf(uniform)
uniform_lccdf = _lccdf(uniform)
uniform_rng = _rng(uniform)

## 22 Distributions over Unbounded Vectors

## 22.1 Multivariate Normal Distribution

# real multi_normal_lpdf(vectors y | vectors mu, matrix Sigma)
# The log of the multivariate normal density of vector(s) y given location vector(s) mu and covariance matrix Sigma

multi_normal = d.MultivariateNormal
multi_normal_lpdf = _lpdf(multi_normal)
multi_normal_rng = _rng(multi_normal)


## 23 Simplex Distributions

## 23.1 Dirichlet Distribution

# real dirichlet_lpdf(vector theta | vector alpha)
# The log of the Dirichlet density for simplex theta given prior counts (plus one) alpha

dirichlet = d.Dirichlet
dirichlet_lpdf = _lpdf(dirichlet)
dirichlet_rng = _rng(dirichlet)

