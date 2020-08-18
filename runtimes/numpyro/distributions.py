import numpyro.distributions as d
from numpyro.distributions import constraints as constraints
import jax.numpy as jnp

class improper_uniform(d.Normal):
    def __init__(self, shape=None):
        zeros = jnp.zeros(shape) if shape else 0
        ones = jnp.ones(shape) if shape else 1
        super(improper_uniform, self).__init__(zeros, ones)

    def log_prob(self, x):
        return jnp.zeros_like(x)


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
