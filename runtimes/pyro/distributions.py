import pyro.distributions as d
from torch.distributions import constraints, transform_to as transform
from pyro.distributions.constraints import Constraint
from numbers import Number
from torch import (
    norm as tnorm,
    log as tlog,
    exp as texp,
    matmul as tmatmul,
    ones as tones,
    zeros as tzeros,
    zeros_like as tzeros_like,
    long as dtype_long,
    float as dtype_float,
    tensor,
    stack
)
import torch

def tsort(x):
    return torch.sort(x).values

def array(x, dtype=None):
  if isinstance(x, list):
    return stack([tensor(e, dtype=dtype) for e in x])
  return tensor(x, dtype=dtype)

d.BernoulliLogits = lambda logits: d.Bernoulli(logits=logits)
d.BinomialLogits = lambda logits, n: d.Binomial(n, logits=logits)
d.Logistic = d.LogisticNormal

def _XXX_TODO_XXX_(f):
    def todo(*args):
        assert False, f"{f}: not yet implemented"

    return todo

def _cast_float(x):
    if isinstance(x, Number):
        return array(x, dtype=dtype_float)
    return x.type(dtype_float)


## Utility functions
def _cast1(f):
    def f_casted(y, *args):
        return f(_cast_float(y), *args)

    return f_casted


def _distrib(d, nargs, typ):
    def d_ext(*args):
        if len(args) <= nargs:
            return d(*args)
        else:
            return d(args[0] * tones(args[nargs], dtype=typ), *args[1:nargs])

    return d_ext


def _lpdf(d):
    def lpdf(y, *args):
        return d(*args).log_prob(y)

    return lpdf


_lupdf = _lpdf


def _lpmf(d):
    def lpmf(y, *args):
        return d(*args).log_prob(y)

    return lpmf


_lupmf = _lpmf


def _cdf(d):
    def lccdf(y, *args):
        return d(*args).cdf(y)

    return lccdf


def _lcdf(d):
    def lccdf(y, *args):
        return tlog(d(*args).cdf(y))

    return lccdf


def _lccdf(d):
    def lccdf(y, *args):
        return tlog(1 - d(*args).cdf(y))

    return lccdf


def _rng(d):
    def rng(*args):
        return d(*args).sample()

    return rng


## Priors


class improper_uniform(d.Normal):
    def __init__(self, shape=[]):
        zeros = tzeros(shape) if shape != [] else 0
        ones = tones(shape) if shape != [] else 1
        super(improper_uniform, self).__init__(zeros, ones)

    def log_prob(self, x):
        return tzeros_like(x)


class lower_constrained_improper_uniform(improper_uniform):
    def __init__(self, lower_bound=0, shape=[]):
        super().__init__(shape)
        self.support = constraints.greater_than(lower_bound)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform(self.support)(s)


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
        fmt_string += "(upper_bound={})".format(self.upper_bound)
        return fmt_string


less_than_eq = _LessThanEq


class upper_constrained_improper_uniform(improper_uniform):
    def __init__(self, upper_bound=0, shape=[]):
        super().__init__(shape)
        self.support = less_than_eq(upper_bound)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform(self.support)(s)


class simplex_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=[]):
        super().__init__(shape)
        self.support = constraints.simplex

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform(self.support)(s)


class unit_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=[]):
        super().__init__(shape)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return s / tnorm(s)


class ordered_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=[]):
        super().__init__(shape)

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return tsort(s)


class positive_ordered_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=[]):
        super().__init__(shape)
        self.support = constraints.positive

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        s = transform(self.support)(s)
        return tsort(s)


class cholesky_factor_corr_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=[]):
        super().__init__(shape[0])
        self.support = constraints.lower_cholesky

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform(self.support)(s)


class cholesky_factor_cov_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=[]):
        super().__init__(shape[0])
        self.support = "XXX TODO XXX"
        assert (
            False
        ), "cholesky_factor_cov_constrained_improper_uniform: not yet implemented"

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform(self.support)(s)


class cov_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=[]):
        super().__init__(shape[0])
        self.support = "XXX TODO XXX"
        assert False, "cov_constrained_improper_uniform: not yet implemented"

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform(self.support)(s)


class corr_constrained_improper_uniform(improper_uniform):
    def __init__(self, shape=[]):
        super().__init__(shape[0])
        self.support = "XXX TODO XXX"
        assert False, "corr_constrained_improper_uniform: not yet implemented"

    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        return transform(self.support)(s)


## Stan distributions

## 12 Binary Distributions

## 12.1 Bernoulli Distribution

# real bernoulli_lpmf(ints y | reals theta)
# The log Bernoulli probability mass of y given chance of success theta

bernoulli = _distrib(d.Bernoulli, 1, dtype_float)
bernoulli_lpmf = _cast1(_lpmf(bernoulli))
bernoulli_lupmf = _cast1(_lupmf(bernoulli))
bernoulli_cdf = _cast1(_cdf(bernoulli))
bernoulli_lcdf = _cast1(_lcdf(bernoulli))
bernoulli_lccdf = _cast1(_lccdf(bernoulli))
bernoulli_rng = _rng(bernoulli)

## 12.2 Bernoulli Distribution, Logit Parameterization

# real bernoulli_logit_lpmf(ints y | reals alpha)
# The log Bernoulli probability mass of y given chance of success inv_logit(alpha)

bernoulli_logit = _distrib(d.BernoulliLogits, 1, dtype_float)
bernoulli_logit_lpmf = _cast1(_lpmf(bernoulli_logit))

## 12.3 Bernoulli-logit generalized linear model (Logistic Regression)

# real bernoulli_logit_glm_lpmf(int y | matrix x, real alpha, vector beta)
# The log Bernoulli probability mass of y given chance of success inv_logit(alpha + x * beta).

bernoulli_logit_glm = lambda x, alpha, beta: bernoulli_logit(alpha + tmatmul(x, beta))
bernoulli_logit_glm_lpmf = _cast1(_lpmf(bernoulli_logit_glm))
bernoulli_logit_glm_lupmf = _cast1(_lupmf(bernoulli_logit_glm))

## 13 Bounded Discrete Distributions

## 13.1 Binomial distribution

# real binomial_lpmf(ints n | ints N, reals theta)
# The log binomial probability mass of n successes in N trials given chance of success theta

binomial = _XXX_TODO_XXX_("binomial")
binomial_lpmf = _lpmf(binomial)
binomial_lupmf = _lupmf(binomial)
binomial_cdf = _cdf(binomial)
binomial_lcdf = _lcdf(binomial)
binomial_lccdf = _lccdf(binomial)
binomial_rng = _rng(binomial)

## 13.2 Binomial Distribution, Logit Parameterization

# real binomial_logit_lpmf(ints n | ints N, reals alpha)
# The log binomial probability mass of n successes in N trials given logit-scaled chance of success alpha

binomial_logit = _distrib(lambda n, logits: d.BinomialLogits(logits, n), 2, dtype_long)
binomial_logit_lpmf = _cast1(_lpmf(binomial_logit))
binomial_logit_lupmf = _cast1(_lupmf(binomial_logit))

## 13.3 Beta-binomial distribution

# real beta_binomial_lpmf(ints n | ints N, reals alpha, reals beta)
# The log beta-binomial probability mass of n successes in N trials given prior success count (plus one) of alpha and prior failure count (plus one) of beta

beta_binomial = _XXX_TODO_XXX_("beta_binomial")
beta_binomial_lpmf = _lpmf(beta_binomial)
beta_binomial_lupmf = _lupmf(beta_binomial)
beta_binomial_cdf = _cdf(beta_binomial)
beta_binomial_lcdf = _lcdf(beta_binomial)
beta_binomial_lccdf = _lccdf(beta_binomial)
beta_binomial_rng = _rng(beta_binomial)

## 13.4 Hypergeometric distribution

# real hypergeometric_lpmf(int n | int N, int a, int b)
# The log hypergeometric probability mass of n successes in N trials given total success count of a and total failure count of b

hypergeometric = _XXX_TODO_XXX_("hypergeometric")
hypergeometric_lpmf = _lpmf(hypergeometric)
hypergeometric_lupmf = _lupmf(hypergeometric)
hypergeometric_rng = _rng(hypergeometric)

## 13.5 Categorical Distribution

# real categorical_lpmf(ints y | vector theta)
# The log categorical probability mass function with outcome(s) y in 1:N given N-vector of outcome probabilities theta. The parameter theta must have non-negative entries that sum to one, but it need not be a variable declared as a simplex.

categorical = _distrib(d.Categorical, 1, dtype_float)
categorical_lpmf = lambda y, theta: _lpmf(categorical)(y - 1, theta)
categorical_lupmf = lambda y, theta: _lupmf(categorical)(y - 1, theta)
categorical_rng = lambda theta: _rng(categorical)(theta) + 1

# real categorical_logit_lpmf(ints y | vector beta)
# The log categorical probability mass function with outcome(s) y in 1:N
# given log-odds of outcomes beta.

categorical_logit = _distrib(
    lambda logits: d.Categorical(logits=logits), 1, dtype_float
)
categorical_logit_lpmf = lambda y, beta: _lpmf(categorical_logit)(y - 1, beta)
categorical_logit_lupmf = lambda y, beta: _lupmf(categorical_logit)(y - 1, beta)
categorical_logit_rng = lambda beta: _rng(categorical_logit)(beta) + 1

## 13.6 Categorical logit generalized linear model (softmax regression)

# real categorical_logit_glm_lpmf(int y | row_vector x, vector alpha, matrix beta)
# The log categorical probability mass function with outcome y in 1:N given N-vector of log-odds of outcomes alpha + x * beta.

categorical_logit_glm = _XXX_TODO_XXX_("categorical_logit_glm")
categorical_logit_glm_lpmf = lambda y, x, alpha, beta: _lpmf(categorical_logit_glm)(y - 1, x, alpha, beta)
categorical_logit_glm_lupmf = lambda y, x, alpha, beta: _lupmf(categorical_logit_glm)(y - 1, x, alpha, beta)

## 13.7 Discrete range distribution

# real discrete_range_lpmf(ints y | ints l, ints u)
# The log probability mass function with outcome(s) y in l:u.

discrete_range = _XXX_TODO_XXX_("discrete_range")
discrete_range_lpmf = _lpmf(discrete_range)
discrete_range_lupmf = _lupmf(discrete_range)
discrete_range_cdf = _cdf(discrete_range)
discrete_range_lcdf = _lcdf(discrete_range)
discrete_range_lccdf = _lccdf(discrete_range)
discrete_range_rng = _rng(discrete_range)

## 13.8 Ordered logistic distribution

# real ordered_logistic_lpmf(ints k | vector eta, vectors c)
# The log ordered logistic probability mass of k given linear predictors eta, and cutpoints c.

ordered_logistic = _XXX_TODO_XXX_("ordered_logistic")
ordered_logistic_lpmf = lambda k, eta, c: _lpmf(ordered_logistic)(k - 1, eta, c)
ordered_logistic_lupmf = lambda k, eta, c: _lupmf(ordered_logistic)(k - 1, eta, c)
ordered_logistic_rng = lambda eta, c: _rng(ordered_logistic)(eta, c) + 1

## 13.9 Ordered logistic generalized linear model (ordinal regression)

# real ordered_logistic_glm_lpmf(int y | row_vector x, vector beta, vector c)
# The log ordered logistic probability mass of y, given linear predictors x * beta, and cutpoints c. The cutpoints c must be ordered.

ordered_logistic_glm = _XXX_TODO_XXX_("ordered_logistic_glm")
ordered_logistic_glm_lpmf = lambda y, x, beta, c: _lpmf(ordered_logistic_glm)(y - 1, x, beta, c)
ordered_logistic_glm_lupmf = lambda y, x, beta, c: _lupmf(ordered_logistic_glm)(y - 1, x, beta, c)

## 13.10 Ordered probit distribution

# real ordered_probit_lpmf(ints k | vector eta, vectors c)
# The log ordered probit probability mass of k given linear predictors eta, and cutpoints c.

ordered_probit = _XXX_TODO_XXX_("ordered_probit")
ordered_probit_lpmf = lambda k, eta, c: _lpmf(ordered_probit)(k - 1, eta, c)
ordered_probit_lupmf = lambda k, eta, c: _lupmf(ordered_probit)(k - 1, eta, c)
ordered_probit_rng = lambda eta, c: _rng(ordered_probit)(eta, c) + 1

## 14 Unbounded Discrete Distributions

## 14.1 Negative binomial distribution

# real neg_binomial_lpmf(ints n | reals alpha, reals beta)
# The log negative binomial probability mass of n given shape alpha and inverse scale beta

neg_binomial = _XXX_TODO_XXX_("neg_binomial")
neg_binomial_lpmf = _cast1(_lpmf(neg_binomial))
neg_binomial_lupmf = _cast1(_lupmf(neg_binomial))
neg_binomial_cdf = _cast1(_cdf(neg_binomial))
neg_binomial_lcdf = _cast1(_lcdf(neg_binomial))
neg_binomial_lccdf = _cast1(_lccdf(neg_binomial))
neg_binomial_rng = _rng(neg_binomial)

## 14.2 Negative Binomial Distribution (alternative parameterization)

# real neg_binomial_2_lpmf(ints n | reals mu, reals phi)
# The negative binomial probability mass of n given location mu and precision phi.

neg_binomial_2 = _distrib(d.GammaPoisson, 2, dtype_float)
neg_binomial_2_lpmf = _cast1(_lpmf(neg_binomial_2))
neg_binomial_2_lupmf = _cast1(_lupmf(neg_binomial_2))
neg_binomial_2_cdf = _cast1(_cdf(neg_binomial_2))
neg_binomial_2_lcdf = _cast1(_lcdf(neg_binomial_2))
neg_binomial_2_lccdf = _cast1(_lccdf(neg_binomial_2))
neg_binomial_2_rng = _rng(neg_binomial_2)

## 14.3 Negative binomial distribution (log alternative parameterization)

# real neg_binomial_2_log_lpmf(ints n | reals eta, reals phi)
# The log negative binomial probability mass of n given log-location eta and inverse overdispersion parameter phi.

neg_binomial_2_log = _XXX_TODO_XXX_("neg_binomial_2_log")
neg_binomial_2_log_lpmf = _cast1(_lpmf(neg_binomial_2_log))
neg_binomial_2_log_lupmf = _cast1(_lupmf(neg_binomial_2_log))
neg_binomial_2_log_rng = _rng(neg_binomial_2_log)

## 14.4 Negative-binomial-2-log generalized linear model (negative binomial regression)

# real neg_binomial_2_log_glm_lpmf(int y | matrix x, real alpha, vector beta, real phi)
# The log negative binomial probability mass of y given log-location alpha + x * beta and inverse overdispersion parameter phi.

neg_binomial_2_log_glm = lambda x, alpha, beta, phi: neg_binomial_2_log(alpha + tmatmul(x, beta), phi)
neg_binomial_2_log_glm_lpmf = _cast1(_lpmf(neg_binomial_2_log_glm))
neg_binomial_2_log_glm_lupmf = _cast1(_lupmf(neg_binomial_2_log_glm))

## 14.5 Poisson Distribution

# real poisson_lpmf(ints n | reals lambda)
# The log Poisson probability mass of n given rate lambda

poisson = _distrib(d.Poisson, 1, dtype_float)
poisson_lpmf = _cast1(_lpmf(poisson))
poisson_lupmf = _cast1(_lupmf(poisson))
poisson_cdf = _cast1(_cdf(poisson))
poisson_lcdf = _cast1(_lcdf(poisson))
poisson_lccdf = _cast1(_lccdf(poisson))
poisson_rng = _rng(poisson)

## 14.6 Poisson Distribution, Log Parameterization

# real poisson_log_lpmf(ints n | reals alpha)
# The log Poisson probability mass of n given log rate alpha

poisson_log = _distrib(lambda alpha: d.Poisson(texp(alpha)), 1, dtype_float)
poisson_log_lpmf = _lpmf(poisson_log)
poisson_log_lupmf = _lupmf(poisson_log)
poisson_log_rng = _rng(poisson_log)

## 14.7 Poisson-log generalized linear model (Poisson regression)

# real poisson_log_glm_lpmf(int y | matrix x, real alpha, vector beta)
# The log Poisson probability mass of y given the log-rate alpha + x * beta.

poisson_log_glm = lambda x, alpha, beta: poisson_log(alpha + tmatmul(x, beta))
poisson_log_glm_lpmf = _lpmf(poisson_log_glm)
poisson_log_glm_lpmf = _lupmf(poisson_log_glm)

## 15 Multivariate Discrete Distributions

## 15.1 Multinomial distribution

# real multinomial_lpmf(int[] y | vector theta)
# The log multinomial probability mass function with outcome array y of size K given the K-simplex distribution parameter theta and (implicit) total count N = sum(y)

multinomial = _XXX_TODO_XXX_("multinomial")
multinomial_lpmf = _lpmf(multinomial)
multinomial_lupmf = _lupmf(multinomial)
multinomial_rng = _rng(multinomial)

## 15.2 Multinomial distribution, logit parameterization

# real multinomial_logit_lpmf(int[] y | vector theta)
# The log multinomial probability mass function with outcome array y of size K given the K-simplex distribution parameter softmax−1(θ) and (implicit) total count N = sum(y)

multinomial_logit = _XXX_TODO_XXX_("multinomial_logit")
multinomial_logit_lpmf = _lpmf(multinomial_logit)
multinomial_logit_lupmf = _lupmf(multinomial_logit)
multinomial_logit_rng = _rng(multinomial_logit)


## 16 Unbounded Continuous Distributions

# 16.1 Normal Distribution

# real normal_lpdf(reals y | reals mu, reals sigma)
# The log of the normal density of y given location mu and scale sigma

normal = _distrib(d.Normal, 2, dtype_float)
normal_lpdf = _lpdf(normal)
normal_lupdf = _lupdf(normal)
normal_cdf = _cdf(normal)
normal_lcdf = _lcdf(normal)
normal_lccdf = _lccdf(normal)
normal_rng = _rng(normal)

# real std_normal_lpdf(reals y)
# The standard normal (location zero, scale one) log probability density of y.

# std_normal = lambda : d.Normal(0,1)
def std_normal(*args):
    if len(args) > 0:
        return d.Normal(0, tones(args[0]))
    else:
        return d.Normal(0, 1)


std_normal_lpdf = _lpdf(std_normal)
std_normal_lupdf = _lupdf(std_normal)
std_normal_cdf = _cdf(std_normal)
std_normal_lcdf = _lcdf(std_normal)
std_normal_lccdf = _lccdf(std_normal)
std_normal_rng = _rng(std_normal)

## 16.2 Normal-id generalized linear model (linear regression)

# real normal_id_glm_lpdf(real y | matrix x, real alpha, vector beta, real sigma)
# The log normal probability density of y given location alpha + x * beta and scale sigma.

normal_id_glm = lambda x, alpha, beta, sigma: normal(alpha + tmatmul(x, beta), sigma)
normal_id_glm_lpdf = _lpmf(normal_id_glm)
normal_id_glm_lupdf = _lupmf(normal_id_glm)

## 16.5 Student-T Distribution

# real student_t_lpdf(reals y | reals nu, reals mu, reals sigma)
# The log of the Student-t density of y given degrees of freedom nu, location mu, and scale sigma

student_t = _distrib(d.StudentT, 3, dtype_float)
student_t_lpdf = _lpdf(student_t)
student_t_cdf = _cdf(student_t)
student_t_lcdf = _lcdf(student_t)
student_t_lccdf = _lccdf(student_t)
student_t_rng = _rng(student_t)

## 16.6 Cauchy Distribution

# real cauchy_lpdf(reals y | reals mu, reals sigma)
# The log of the Cauchy density of y given location mu and scale sigma

cauchy = _distrib(d.Cauchy, 2, dtype_float)
cauchy_lpdf = _lpdf(cauchy)
cauchy_cdf = _cdf(cauchy)
cauchy_lcdf = _lcdf(cauchy)
cauchy_lccdf = _lccdf(cauchy)
cauchy_rng = _rng(cauchy)

## 16.7 Double Exponential (Laplace) Distribution

# real double_exponential_lpdf(reals y | reals mu, reals sigma)
# The log of the double exponential density of y given location mu and scale sigma

double_exponential = _distrib(d.Laplace, 2, dtype_float)
double_exponential_lpdf = _lpdf(double_exponential)
double_exponential_cdf = _cdf(double_exponential)
double_exponential_lcdf = _lcdf(double_exponential)
double_exponential_lccdf = _lccdf(double_exponential)
double_exponential_rng = _rng(double_exponential)

## 16.8 Logistic Distribution

# real logistic_lpdf(reals y | reals mu, reals sigma)
# The log of the logistic density of y given location mu and scale sigma

logistic = _distrib(d.Logistic, 2, dtype_float)
logistic_lpdf = _lpdf(logistic)
logistic_cdf = _cdf(logistic)
logistic_lcdf = _lcdf(logistic)
logistic_lccdf = _lccdf(logistic)
logistic_rng = _rng(logistic)


## 17 Positive Continuous Distributions

## 17.1 Lognormal Distribution

# real lognormal_lpdf(reals y | reals mu, reals sigma)
# The log of the lognormal density of y given location mu and scale sigma

lognormal = _distrib(d.LogNormal, 2, dtype_float)
lognormal_lpdf = _lpdf(lognormal)
lognormal_cdf = _cdf(lognormal)
lognormal_lcdf = _lcdf(lognormal)
lognormal_lccdf = _lccdf(lognormal)
lognormal_rng = _rng(lognormal)

## 17.5 Exponential Distribution

# real exponential_lpdf(reals y | reals beta)
# The log of the exponential density of y given inverse scale beta

exponential = _distrib(d.Exponential, 1, dtype_float)
exponential_lpdf = _lpdf(exponential)
exponential_cdf = _cdf(exponential)
exponential_lcdf = _lcdf(exponential)
exponential_lccdf = _lccdf(exponential)
exponential_rng = _rng(exponential)

## 17.6 Gamma Distribution

# real gamma_lpdf(reals y | reals alpha, reals beta)
# The log of the gamma density of y given shape alpha and inverse scale beta

gamma = _distrib(d.Gamma, 2, dtype_float)
gamma_lpdf = _lpdf(gamma)
gamma_cdf = _cdf(gamma)
gamma_lcdf = _lcdf(gamma)
gamma_lccdf = _lccdf(gamma)
gamma_rng = _rng(gamma)

## 17.7 Inverse Gamma Distribution

# real inv_gamma_lpdf(reals y | reals alpha, reals beta)
# The log of the inverse gamma density of y given shape alpha and scale beta

inv_gamma = _distrib(d.InverseGamma, 2, dtype_float)
inv_gamma_lpdf = _lpdf(inv_gamma)
inv_gamma_cdf = _cdf(inv_gamma)
inv_gamma_lcdf = _lcdf(inv_gamma)
inv_gamma_lccdf = _lccdf(inv_gamma)
inv_gamma_rng = _rng(inv_gamma)

## 18 Positive Lower-Bounded Distributions

## 18.1 Pareto Distribution

# real pareto_lpdf(reals y | reals y_min, reals alpha)
# The log of the Pareto density of y given positive minimum value y_min and shape alpha

pareto = _distrib(d.Pareto, 2, dtype_float)
pareto_lpdf = _lpdf(pareto)
pareto_cdf = _cdf(pareto)
pareto_lcdf = _lcdf(pareto)
pareto_lccdf = _lccdf(pareto)
pareto_rng = _rng(pareto)

## 19 Continuous Distributions on [0, 1]

## 19.1 Beta Distribution

# real beta_lpdf(reals theta | reals alpha, reals beta)
# The log of the beta density of theta in [0,1] given positive prior
# successes (plus one) alpha and prior failures (plus one) beta

beta = _distrib(d.Beta, 2, dtype_float)
beta_lpdf = _lpdf(beta)
beta_cdf = _cdf(beta)
beta_lcdf = _lcdf(beta)
beta_lccdf = _lccdf(beta)
beta_rng = _rng(beta)


## 21 Bounded Continuous Probabilities

## 21.1 Uniform Distribution

# real uniform_lpdf(reals y | reals alpha, reals beta)
# The log of the uniform density of y given lower bound alpha and upper bound beta

uniform = _distrib(d.Uniform, 2, dtype_float)
uniform_lpdf = _lpdf(uniform)
uniform_cdf = _cdf(uniform)
uniform_lcdf = _lcdf(uniform)
uniform_lccdf = _lccdf(uniform)
uniform_rng = _rng(uniform)

## 22 Distributions over Unbounded Vectors

## 22.1 Multivariate Normal Distribution

# real multi_normal_lpdf(vectors y | vectors mu, matrix Sigma)
# The log of the multivariate normal density of vector(s) y given location vector(s) mu and covariance matrix Sigma

multi_normal = _distrib(d.MultivariateNormal, 2, dtype_float)
multi_normal_lpdf = _lpdf(multi_normal)
multi_normal_rng = _rng(multi_normal)

## 22.3 Multivariate Normal Distribution, Cholesky Parameterization

# real multi_normal_cholesky_lpdf(vectors y | vectors mu, matrix L)
# The log of the multivariate normal density of vector(s) y given location vector(s) mu and lower-triangular Cholesky factor of the covariance matrix L

multi_normal_cholesky = _distrib(
    lambda mu, l: d.MultivariateNormal(mu, tmatmul(l, l.t())), 2, dtype_float
)
multi_normal_cholesky_lpdf = _lpdf(multi_normal_cholesky)
multi_normal_cholesky_rng = _rng(multi_normal_cholesky)

## 23 Simplex Distributions

## 23.1 Dirichlet Distribution

# real dirichlet_lpdf(vector theta | vector alpha)
# The log of the Dirichlet density for simplex theta given prior counts (plus one) alpha

dirichlet = _distrib(d.Dirichlet, 1, dtype_float)
dirichlet_lpdf = _lpdf(dirichlet)
dirichlet_rng = _rng(dirichlet)
