import torch
from torch import ones, tensor
from numbers import Number

def dispatch1(tf, mf, x):
    if isinstance(x, torch.Tensor):
        return tf(x)
    else:
        return mf(x)

def _XXX_TODO_XXX_(x):
    assert false, f'{x}: not yet implemented'

## 3.8 Power and Logarithm Functions

from torch import sqrt as tsqrt
from math import sqrt as msqrt
def sqrt(x):
    """
    R sqrt(T x)
    square root of x
    """
    return dispatch1(tsqrt, msqrt, x)

# R cbrt(T x)
# cube root of x
def cbrt(x):
    _XXX_TODO_XXX_('cbrt')

from torch import square as tsquare
def square(x):
    """
    R square(T x)
    square of x
    """
    return dispatch1(tsquare, lambda x: x * x, x)

from torch import exp as texp
from math import exp as mexp
def exp(x):
    """
    R exp(T x)
    natural exponential of x
    """
    return dispatch1(texp, mexp, x)

# R exp2(T x)
# base-2 exponential of x
def exp2(x):
    _XXX_TODO_XXX_('exp2')

from torch import log as tlog
from math import log as mlog
def log(x):
    """
    R log(T x)
    natural logarithm of x
    """
    return dispatch1(tlog, mlog, x)

from torch import log2 as tlog2
from math import log2 as mlog2
def log2(x):
    """
    R log2(T x)
    base-2 logarithm of x
    """
    return dispatch1(tlog2, mlog2, x)

from torch import log10 as tlog10
from math import log10 as mlog10
def log10(x):
    """
    R log10(T x)
    base-10 logarithm of x
    """
    return dispatch1(tlog10, mlog10, x)

def pow(x, y):
    """
    real pow(real x, real y)
    Return x raised to the power of y.
    """
    return x ** y

# R inv(T x)
# inverse of x
def inv(x):
    _XXX_TODO_XXX_('inv')

# R inv_sqrt(T x)
# inverse of the square root of x
def inv_sqrt(x):
    _XXX_TODO_XXX_('inv_sqrt')

# R inv_square(T x)
# inverse of the square of x
def inv_square(x):
    _XXX_TODO_XXX_('inv_square')

## 4.1 Reductions

# 4.1.1 Minimum and Maximum

# real min(real[] x)
# The minimum value in x, or +∞  if x is size 0.
# int min(int[] x)
# The minimum value in x, or error if x is size 0.
from torch import min

# real max(real[] x)
# The maximum value in x, or −∞ if x is size 0.
# int max(int[] x)
# The maximum value in x, or error if x is size 0.
from torch import max


## 4.1.2 Sum, Product, and Log Sum of Exp

# int sum(int[] x)
# The sum of the elements in x, defined for x
# of size N by sum(x)
# real sum(real[] x)
# The sum of the elements in x; see definition above.
from torch import sum

# real prod(real[] x)
# The product of the elements in x, or 1 if x is size 0.
# real prod(int[] x)
# The product of the elements in x, product(x)={∏Nn=1xnifN>01ifN=0
from torch import prod

# real log_sum_exp(real[] x)
# The natural logarithm of the sum of the exponentials of the elements in x, or −∞
# if the array is empty.
from torch import logsumexp as log_sum_exp

## 4.1.3 Sample Mean, Variance, and Standard Deviation

# real mean(real[] x)
# The sample mean of the elements in x.
# It is an error to the call the mean function with an array of size 0.
from torch import mean

# real variance(real[] x)
# The sample variance of the elements in x.
# It is an error to call the variance function with an array of size 0.
from torch import var as variance

# real sd(real[] x)
# The sample standard deviation of elements in x.
# It is an error to call the sd function with an array of size 0.
from torch import std as sd


## 4.1.4 Euclidean Distance and Squared Distance

# real distance(vector x, vector y)
# real distance(vector x, row_vector y)
# real distance(row_vector x, vector y)
# real distance(row_vector x, row_vector y)
# The Euclidean distance between x and y
def distance(x):
    _XXX_TODO_XXX_('distance')

# real squared_distance(vector x, vector y)
# real squared_distance(vector x, row_vector [] y)
# real squared_distance(row_vector x, vector [] y)
# real squared_distance(row_vector x, row_vector[] y)
# The squared Euclidean distance between x and y
def squared_distance(x):
    _XXX_TODO_XXX_('squared_distance')


## 5.7 Broadcast Functions

def rep_vector(x, m):
    """
    vector rep_vector(real x, int m)
    Return the size m (column) vector consisting of copies of x.
    """
    return x * ones(m)

def rep_row_vector(x, m):
    """
    row_vector rep_row_vector(real x, int n)
    Return the size n row vector consisting of copies of x.
    """
    return x * ones(1, m)

def rep_matrix(x, *dims):
    """
    matrix rep_matrix(real x, int m, int n)
    Return the m by n matrix consisting of copies of x.
    matrix rep_matrix(vector v, int n)
    Return the m by n matrix consisting of n copies of the (column) vector v of size m.
    matrix rep_matrix(row_vector rv, int m)
    Return the m by n matrix consisting of m copies of the row vector rv of size n.
    """
    if len(dims) == 2:
        return x * ones(*dims)
    elif len(x.shape) == 1:
        return x * ones(x.shape[0], *dims)
    elif len(x.shape) == 2:
        return x * ones(*dims, x.shape[1])
    else:
        assert False

## 5.10 Matrix Concatenation

# Horizontal concatenation
from torch import cat
def append_col(x, y):
    """
    matrix append_col(matrix x, matrix y)
    Combine matrices x and y by columns. The matrices must have the same number of rows.
    matrix append_col(matrix x, vector y)
    Combine matrix x and vector y by columns. The matrix and the vector must have the same number of rows.
    matrix append_col(vector x, matrix y)
    Combine vector x and matrix y by columns. The vector and the matrix must have the same number of rows.
    matrix append_col(vector x, vector y)
    Combine vectors x and y by columns. The vectors must have the same number of rows.
    row_vector append_col(row_vector x, row_vector y)
    Combine row vectors x and y of any size into another row vector.
    row_vector append_col(real x, row_vector y)
    Append x to the front of y, returning another row vector.
    row_vector append_col(row_vector x, real y)
    Append y to the end of x, returning another row vector.
    """
    # XXX TODO: review XXX
    if isinstance(x, Number):
        return cat([x * ones([1,1], dtype=torch.double), y])
    elif isinstance(y, Number):
        return cat([x, y * ones([1,1], dtype=torch.double)])
    elif len(x.shape) == 1 and len(y.shape) == 1:
        return cat([x.expand([1,x.shape[0]]), x.expand([1,x.shape[0]])], 1)
    else:
        return cat([x, y], 1)

# 5.10.0.2 Vertical concatenation

def append_row(x, y):
    """
    matrix append_row(matrix x, matrix y)
    Combine matrices x and y by rows. The matrices must have the same number of columns.
    matrix append_row(matrix x, row_vector y)
    Combine matrix x and row vector y by rows. The matrix and the row vector must have the same number of columns.
    matrix append_row(row_vector x, matrix y)
    Combine row vector x and matrix y by rows. The row vector and the matrix must have the same number of columns.
    matrix append_row(row_vector x, row_vector y)
    Combine row vectors x and y by row. The row vectors must have the same number of columns.
    vector append_row(vector x, vector y)
    Concatenate vectors x and y of any size into another vector.
    vector append_row(real x, vector y)
    Append x to the top of y, returning another vector.
    vector append_row(vector x, real y)
    Append y to the bottom of x, returning another vector.
    """
    # XXX TODO: review XXX
    if isinstance(x, Number):
        return cat([tensor([x], dtype=torch.double), y])
    elif isinstance(y, Number):
        return cat([x, tensor([y], dtype=torch.double)])
    else:
        return cat([x, y])
