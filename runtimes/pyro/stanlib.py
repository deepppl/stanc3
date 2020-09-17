import torch
from torch import ones

def dispatch1(tf, mf, x):
    if isinstance(x, torch.Tensor):
        return tf(x)
    else:
        return mf(x)

from torch import mean


## Power and Logarithm Functions (3.8)

# R sqrt(T x)
# square root of x
from torch import sqrt as tsqrt
from math import sqrt as msqrt
def sqrt(x):
    return dispatch1(tsqrt, msqrt, x)

# R cbrt(T x)
# cube root of x
# XXX TODO XXX

# R square(T x)
# square of x
from torch import square as tsquare
def msquare(x):
    return x * x
def square(x):
    return dispatch1(tsquare, msquare, x)

# R exp(T x)
# natural exponential of x
from torch import exp as texp
from math import exp as mexp
def exp(x):
    return dispatch1(texp, mexp, x)

# R exp2(T x)
# base-2 exponential of x
# XXX TODO XXX

# R log(T x)
# natural logarithm of x
from torch import log as tlog
from math import log as mlog
def log(x):
    return dispatch1(tlog, mlog, x)

# R log2(T x)
# base-2 logarithm of x
from torch import log2 as tlog2
from math import log2 as mlog2
def log2(x):
    return dispatch1(tlog2, mlog2, x)

# R log10(T x)
# base-10 logarithm of x
from torch import log10 as tlog10
from math import log10 as mlog10
def log10(x):
    return dispatch1(tlog10, mlog10, x)

# real pow(real x, real y)
# Return x raised to the power of y.
# XXX TODO XXX

# R inv(T x)
# inverse of x
# XXX TODO XXX

# R inv_sqrt(T x)
# inverse of the square root of x
# XXX TODO XXX

# R inv_square(T x)
# inverse of the square of x
# XXX TODO XXX


## 5.7 Broadcast Functions

# vector rep_vector(real x, int m)
# Return the size m (column) vector consisting of copies of x.
def rep_vector(x, m):
    return x * ones(m)

# row_vector rep_row_vector(real x, int n)
# Return the size n row vector consisting of copies of x.
def rep_row_vector(x, m):
    return x * ones(1, m)

# matrix rep_matrix(real x, int m, int n)
# Return the m by n matrix consisting of copies of x.
# matrix rep_matrix(vector v, int n)
# Return the m by n matrix consisting of n copies of the (column) vector v of size m.
# matrix rep_matrix(row_vector rv, int m)
# Return the m by n matrix consisting of m copies of the row vector rv of size n.
def rep_matrix(x, *dims):
    if len(dims) == 2:
        return x * ones(*dims)
    elif len(x.shape) == 1:
        return x.expand(x.shape[0], *dims)
    elif len(x.shape) == 2:
        return x.expand(*dims, x.shape[1])
    else:
        assert False
