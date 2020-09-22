import math
import torch
import torch.nn
from torch import ones, tensor, Tensor

def _XXX_TODO_XXX_(f):
    def todo(x):
        assert false, f'{f}: not yet implemented'
    return todo

# 3.2 Mathematical Constants

# real pi()
# π, the ratio of a circle’s circumference to its diameter
pi = lambda: math.pi

# real e()
# e, the base of the natural logarithm
e = lambda: math.e

# real sqrt2()
# The square root of 2
from math import sqrt as msqrt
sqrt2 = lambda: msqrt(2)

# real log2()
# The natural logarithm of 2
from math import log as mlog
log2 = lambda: mlog(2)

# real log10()
# The natural logarithm of 10
log10 = lambda: mlog(10)


## 3.8 Power and Logarithm Functions

# R sqrt(T x)
# square root of x
from torch import sqrt as tsqrt
sqrt_int = msqrt
sqrt_real = msqrt
sqrt_vector = tsqrt
sqrt_rowvector = tsqrt
sqrt_matrix = tsqrt
sqrt_array = tsqrt

# R cbrt(T x)
# cube root of x
cbrt_int = _XXX_TODO_XXX_('cbrt')
cbrt_real = _XXX_TODO_XXX_('cbrt')
cbrt_vector = _XXX_TODO_XXX_('cbrt')
cbrt_rowvector = _XXX_TODO_XXX_('cbrt')
cbrt_matrix = _XXX_TODO_XXX_('cbrt')
cbrt_array = _XXX_TODO_XXX_('cbrt')

# R square(T x)
# square of x
from torch import square as tsquare
square_int = lambda x: x * x
square_real = lambda x: x * x
square_vector = tsquare
square_rowvector = tsquare
square_matrix = tsquare
square_array = tsquare


# R exp(T x)
# natural exponential of x
from torch import exp as texp
from math import exp as mexp
exp_int = mexp
exp_real = mexp
exp_vector = texp
exp_rowvector = texp
exp_matrix = texp
exp_array = texp

# R exp2(T x)
# base-2 exponential of x
exp2_int = _XXX_TODO_XXX_('exp2')
exp2_real = _XXX_TODO_XXX_('exp2')
exp2_vector = _XXX_TODO_XXX_('exp2')
exp2_rowvector = _XXX_TODO_XXX_('exp2')
exp2_matrix = _XXX_TODO_XXX_('exp2')
exp2_array = _XXX_TODO_XXX_('exp2')

# R log(T x)
# natural logarithm of x
from torch import log as tlog
log_int = mlog
log_real = mlog
log_vector = tlog
log_rowvector = tlog
log_matrix = tlog
log_array = tlog

# R log2(T x)
# base-2 logarithm of x
from torch import log2 as tlog2
from math import log2 as mlog2
log2_int = mlog2
log2_real = mlog2
log2_vector = tlog2
log2_rowvector = tlog2
log2_matrix = tlog2
log2_array = tlog2

# R log10(T x)
# base-10 logarithm of x
from torch import log10 as tlog10
from math import log10 as mlog10
log10_int = mlog10
log10_real = mlog10
log10_vector = tlog10
log10_rowvector = tlog10
log10_matrix = tlog10
log10_array = tlog10

# real pow(real x, real y)
# Return x raised to the power of y.
pow_int_int = lambda x, y: x ** y
pow_int_real = lambda x, y: x ** y
pow_real_int = lambda x, y: x ** y
pow_real_real = lambda x, y: x ** y

# R inv(T x)
# inverse of x
inv_int = _XXX_TODO_XXX_('inv')
inv_real = _XXX_TODO_XXX_('inv')
inv_vector = _XXX_TODO_XXX_('inv')
inv_rowvector = _XXX_TODO_XXX_('inv')
inv_matrix = _XXX_TODO_XXX_('inv')
inv_array = _XXX_TODO_XXX_('inv')

# R inv_sqrt(T x)
# inverse of the square root of x
inv_sqrt_int = _XXX_TODO_XXX_('inv_sqrt')
inv_sqrt_real = _XXX_TODO_XXX_('inv_sqrt')
inv_sqrt_vector = _XXX_TODO_XXX_('inv_sqrt')
inv_sqrt_rowvector = _XXX_TODO_XXX_('inv_sqrt')
inv_sqrt_matrix = _XXX_TODO_XXX_('inv_sqrt')
inv_sqrt_array = _XXX_TODO_XXX_('inv_sqrt')

# R inv_square(T x)
# inverse of the square of x
inv_square_int = _XXX_TODO_XXX_('inv_square')
inv_square_real = _XXX_TODO_XXX_('inv_square')
inv_square_vector = _XXX_TODO_XXX_('inv_square')
inv_square_rowvector = _XXX_TODO_XXX_('inv_square')
inv_square_matrix = _XXX_TODO_XXX_('inv_square')
inv_square_array = _XXX_TODO_XXX_('inv_square')

## 3.9 Trigonometric Functions

# real hypot(real x, real y)
# Return the length of the hypotenuse of a right triangle with sides of length x and y.
from math import hypot
hypot_real_real = hypot

# R cos(T x)
# cosine of the angle x (in radians)
from torch import cos as tcos
from math import cos as mcos
cos_int = mcos
cos_real = mcos
cos_vector = tcos
cos_rowvector = tcos
cos_matrix = tcos
cos_array = tcos

# R sin(T x)
# sine of the angle x (in radians)
from torch import sin as tsin
from math import sin as msin
sin_int = msin
sin_real = msin
sin_vector = tsin
sin_rowvector = tsin
sin_matrix = tsin
sin_array = tsin

# R tan(T x)
# tangent of the angle x (in radians)
from torch import tan as ttan
from math import tan as mtan
tan_int = mtan
tan_real = mtan
tan_vector = ttan
tan_rowvector = ttan
tan_matrix = ttan
tan_array = ttan

# R acos(T x)
# principal arc (inverse) cosine (in radians) of x
from torch import acos as tacos
from math import acos as macos
acos_int = macos
acos_real = macos
acos_vector = tacos
acos_rowvector = tacos
acos_matrix = tacos
acos_array = tacos

# R asin(T x)
# principal arc (inverse) sine (in radians) of x
from torch import asin as tasin
from math import asin as masin
asin_int = masin
asin_real = masin
asin_vector = tasin
asin_rowvector = tasin
asin_matrix = tasin
asin_array = tasin

# R atan(T x)
# principal arc (inverse) tangent (in radians) of x, with values from −π
# to π
from torch import atan as tatan
from math import atan as matan
atan_int = matan
atan_real = matan
atan_vector = tatan
atan_rowvector = tatan
atan_matrix = tatan
atan_array = tatan

# real atan2(real y, real x)
# Return the principal arc (inverse) tangent (in radians) of y divided by x
from math import atan2
atan2_real_real = atan2


## 3.10 Hyperbolic Trigonometric Functions

# R cosh(T x)
# hyperbolic cosine of x (in radians)
from torch import cosh as tcosh
from math import cosh as mcosh
cosh_int = mcosh
cosh_real = mcosh
cosh_vector = tcosh
cosh_rowvector = tcosh
cosh_matrix = tcosh
cosh_array = tcosh

# R sinh(T x)
# hyperbolic sine of x (in radians)
from torch import sinh as tsinh
from math import sinh as msinh
sinh_int = msinh
sinh_real = msinh
sinh_vector = tsinh
sinh_rowvector = tsinh
sinh_matrix = tsinh
sinh_array = tsinh

# R tanh(T x)
# hyperbolic tangent of x (in radians)
from torch import tanh as ttanh
from math import tanh as mtanh
tanh_int = mtanh
tanh_real = mtanh
tanh_vector = ttanh
tanh_rowvector = ttanh
tanh_matrix = ttanh
tanh_array = ttanh

# R acosh(T x)
# inverse hyperbolic cosine (in radians)
from torch import acosh as tacosh
from math import acosh as macosh
acosh_int = macosh
acosh_real = macosh
acosh_vector = tacosh
acosh_rowvector = tacosh
acosh_matrix = tacosh
acosh_array = tacosh

# R asinh(T x)
# inverse hyperbolic cosine (in radians)
from torch import asinh as tasinh
from math import asinh as masinh
asinh_int = masinh
asinh_real = masinh
asinh_vector = tasinh
asinh_rowvector = tasinh
asinh_matrix = tasinh
asinh_array = tasinh

# R atanh(T x)
# inverse hyperbolic tangent (in radians) of x
from torch import atanh as tatanh
from math import atanh as matanh
atanh_int = matanh
atanh_real = matanh
atanh_vector = tatanh
atanh_rowvector = tatanh
atanh_matrix = tatanh
atanh_array = tatanh


## 3.11 Link Functions

# R logit(T x)
# log odds, or logit, function applied to x
logit_int = _XXX_TODO_XXX_('logit')
logit_real = _XXX_TODO_XXX_('logit')
logit_vector = _XXX_TODO_XXX_('logit')
logit_rowvector = _XXX_TODO_XXX_('logit')
logit_matrix = _XXX_TODO_XXX_('logit')
logit_array = _XXX_TODO_XXX_('logit')

# R inv_logit(T x)
# logistic sigmoid function applied to x
from torch import sigmoid
inv_logit_int = lambda x: sigmoid(Tensor([x]))
inv_logit_real = lambda x: sigmoid(Tensor([x]))
inv_logit_vector = sigmoid
inv_logit_rowvector = sigmoid
inv_logit_matrix = sigmoid
inv_logit_array = sigmoid

# R inv_cloglog(T x)
# inverse of the complementary log-log function applied to x
inv_cloglog_int = _XXX_TODO_XXX_('inv_cloglog')
inv_cloglog_real = _XXX_TODO_XXX_('inv_cloglog')
inv_cloglog_vector = _XXX_TODO_XXX_('inv_cloglog')
inv_cloglog_rowvector = _XXX_TODO_XXX_('inv_cloglog')
inv_cloglog_matrix = _XXX_TODO_XXX_('inv_cloglog')
inv_cloglog_array = _XXX_TODO_XXX_('inv_cloglog')


## 3.14 Composed Functions

# R expm1(T x)
# natural exponential of x minus 1
expm1_int = _XXX_TODO_XXX_('expm1')
expm1_real = _XXX_TODO_XXX_('expm1')
expm1_vector = _XXX_TODO_XXX_('expm1')
expm1_rowvector = _XXX_TODO_XXX_('expm1')
expm1_matrix = _XXX_TODO_XXX_('expm1')
expm1_array = _XXX_TODO_XXX_('expm1')


# real fma(real x, real y, real z)
# Return z plus the result of x multiplied by y. fma(x,y,z)=(x×y)+z
fma_real_real_real = _XXX_TODO_XXX_('fma')

# real multiply_log(real x, real y)
# Warning: This function is deprecated and should be replaced with lmultiply. Return the product of x and the natural logarithm of y.
multiply_log_real_real = _XXX_TODO_XXX_('multiply_log')

# real lmultiply(real x, real y)
# Return the product of x and the natural logarithm of y.
lmultiply_real_real = _XXX_TODO_XXX_('lmultiply')

# R log1p(T x)
# natural logarithm of 1 plus x
log1p_int = _XXX_TODO_XXX_('log1p')
log1p_real = _XXX_TODO_XXX_('log1p')
log1p_vector = _XXX_TODO_XXX_('log1p')
log1p_rowvector = _XXX_TODO_XXX_('log1p')
log1p_matrix = _XXX_TODO_XXX_('log1p')
log1p_array = _XXX_TODO_XXX_('log1p')

# R log1m(T x)
# natural logarithm of 1 minus x
log1m_int = _XXX_TODO_XXX_('log1m')
log1m_real = _XXX_TODO_XXX_('log1m')
log1m_vector = _XXX_TODO_XXX_('log1m')
log1m_rowvector = _XXX_TODO_XXX_('log1m')
log1m_matrix = _XXX_TODO_XXX_('log1m')
log1m_array = _XXX_TODO_XXX_('log1m')

# R log1p_exp(T x)
# natural logarithm of one plus the natural exponentiation of x
log1p_exp_int = _XXX_TODO_XXX_('log1p_exp')
log1p_exp_real = _XXX_TODO_XXX_('log1p_exp')
log1p_exp_vector = _XXX_TODO_XXX_('log1p_exp')
log1p_exp_rowvector = _XXX_TODO_XXX_('log1p_exp')
log1p_exp_matrix = _XXX_TODO_XXX_('log1p_exp')
log1p_exp_array = _XXX_TODO_XXX_('log1p_exp')

# R log1m_exp(T x)
# logarithm of one minus the natural exponentiation of x
log1m_exp_int = _XXX_TODO_XXX_('log1m_exp')
log1m_exp_real = _XXX_TODO_XXX_('log1m_exp')
log1m_exp_vector = _XXX_TODO_XXX_('log1m_exp')
log1m_exp_rowvector = _XXX_TODO_XXX_('log1m_exp')
log1m_exp_matrix = _XXX_TODO_XXX_('log1m_exp')
log1m_exp_array = _XXX_TODO_XXX_('log1m_exp')

# real log_diff_exp(real x, real y)
# Return the natural logarithm of the difference of the natural exponentiation of x and the natural exponentiation of y.
log_diff_exp_real_real = _XXX_TODO_XXX_('log_diff_exp')

# real log_mix(real theta, real lp1, real lp2)
# Return the log mixture of the log densities lp1 and lp2 with mixing proportion theta, defined by log_mix(θ,λ1,λ2)=log(θexp(λ1)+(1−θ)exp(λ2))=log_sum_exp(log(θ)+λ1, log(1−θ)+λ2).
def log_mix_real_real_real(theta, lp1, lp2):
    return log_sum_exp_real_real(log_real(theta) + lp1, log_real(1 - theta) + lp2)

# real log_sum_exp(real x, real y)
# Return the natural logarithm of the sum of the natural exponentiation of x and the natural exponentiation of y. log_sum_exp(x,y)=log(exp(x)+exp(y))
from torch import logsumexp
def log_sum_exp_real_real(x, y):
    max = x if x > y else y
    dx = x - max
    dy = y - max
    sum_of_exp = exp_real(dx) + exp_real(dy)
    return max + log_real(sum_of_exp)

# R log_inv_logit(T x)
# natural logarithm of the inverse logit function of x
from torch.nn import LogSigmoid
log_inv_logit_int = lambda x: LogSigmoid(Tensor([x]))
log_inv_logit_real = lambda x: LogSigmoid(Tensor([x]))
log_inv_logit_vector = LogSigmoid
log_inv_logit_rowvector = LogSigmoid
log_inv_logit_matrix = LogSigmoid
log_inv_logit_array = LogSigmoid

# R log1m_inv_logit(T x)
# natural logarithm of 1 minus the inverse logit function of x
log1m_inv_logit_int = _XXX_TODO_XXX_('log_inv_logit')
log1m_inv_logit_real = _XXX_TODO_XXX_('log_inv_logit')
log1m_inv_logit_vector = _XXX_TODO_XXX_('log_inv_logit')
log1m_inv_logit_rowvector = _XXX_TODO_XXX_('log_inv_logit')
log1m_inv_logit_matrix = _XXX_TODO_XXX_('log_inv_logit')
log1m_inv_logit_array = _XXX_TODO_XXX_('log_inv_logit')


## 4.1 Reductions

# 4.1.1 Minimum and Maximum

# real min(real[] x)
# The minimum value in x, or +∞  if x is size 0.
# int min(int[] x)
# The minimum value in x, or error if x is size 0.
from torch import min
min_array = min

# real max(real[] x)
# The maximum value in x, or −∞ if x is size 0.
# int max(int[] x)
# The maximum value in x, or error if x is size 0.
from torch import max
max_array = max


## 4.1.2 Sum, Product, and Log Sum of Exp

# int sum(int[] x)
# The sum of the elements in x, defined for x
# of size N by sum(x)
# real sum(real[] x)
# The sum of the elements in x; see definition above.
from torch import sum
sum_array = sum

# real prod(real[] x)
# The product of the elements in x, or 1 if x is size 0.
# real prod(int[] x)
# The product of the elements in x, product(x)={∏Nn=1xnifN>01ifN=0
from torch import prod
prod_array = prod

# real log_sum_exp(real[] x)
# The natural logarithm of the sum of the exponentials of the elements in x, or −∞
# if the array is empty.
log_sum_exp_array = logsumexp

## 4.1.3 Sample Mean, Variance, and Standard Deviation

# real mean(real[] x)
# The sample mean of the elements in x.
# It is an error to the call the mean function with an array of size 0.
from torch import mean
mean_array = mean

# real variance(real[] x)
# The sample variance of the elements in x.
# It is an error to call the variance function with an array of size 0.
from torch import var
variance_array = var

# real sd(real[] x)
# The sample standard deviation of elements in x.
# It is an error to call the sd function with an array of size 0.
from torch import std
sd_array = std


## 4.1.4 Euclidean Distance and Squared Distance

# real distance(vector x, vector y)
# real distance(vector x, row_vector y)
# real distance(row_vector x, vector y)
# real distance(row_vector x, row_vector y)
# The Euclidean distance between x and y
distance_vector_vector = _XXX_TODO_XXX_('distance')
distance_vector_rowvector = _XXX_TODO_XXX_('distance')
distance_rowvector_vector = _XXX_TODO_XXX_('distance')
distance_rowvector_rowvector = _XXX_TODO_XXX_('distance')

# real squared_distance(vector x, vector y)
# real squared_distance(vector x, row_vector [] y)
# real squared_distance(row_vector x, vector [] y)
# real squared_distance(row_vector x, row_vector[] y)
# The squared Euclidean distance between x and y
squared_distance_vector_vector = _XXX_TODO_XXX_('squared_distance')
squared_distance_vector_rowvector = _XXX_TODO_XXX_('squared_distance')
squared_distance_rowvector_vector = _XXX_TODO_XXX_('squared_distance')
squared_distance_rowvector_rowvector = _XXX_TODO_XXX_('squared_distance')


## 4.2 Array Size and Dimension Function

# int[] dims(T x)
# Return an integer array containing the dimensions of x; the type
# of the argument T can be any Stan type with up to 8 array
# dimensions.
dims_int = tensor([], dtype=torch.long)
dims_real = tensor([], dtype=torch.long)
dims_vector = lambda x: tensor(x.shape)
dims_rowvector = lambda x: tensor(x.shape)
dims_matrix = lambda x: tensor(x.shape)
dims_array = lambda x: tensor(x.shape)

# int num_elements(T[] x)
# Return the total number of elements in the array x including all
# elements in contained arrays, vectors, and matrices. T can be any
# array type. For example, if x is of type real[4,3] then
# num_elements(x) is 12, and if y is declared as matrix[3,4] y[5],
# then size(y) evaluates to 60.
num_elements_array = lambda x: math.prod(x.shape)

# int size(T[] x)
# Return the number of elements in the array x; the type of the array T
# can be any type, but the size is just the size of the top level
# array, not the total number of elements contained. For example, if
# x is of type real[4,3] then size(x) is 4.
size_array = lambda x: x.shape[0]

## 5 Matrix Operations

# 5.1 Integer-Valued Matrix Size Functions

# int num_elements(vector x)
# The total number of elements in the vector x (same as function rows)
num_elements_vector = lambda x: x.shape[0]

# int num_elements(row_vector x)
# The total number of elements in the vector x (same as function cols)
num_elements_rowvector = lambda x: x.shape[0]

# int num_elements(matrix x)
# The total number of elements in the matrix x. For example, if x is a 5×3
# matrix, then num_elements(x) is 15
num_elements_matrix = lambda x: x.shape[0] * x.shape[1]

# int rows(vector x)
# The number of rows in the vector x
rows_vector = lambda x: x.shape[0]

# int rows(row_vector x)
# The number of rows in the row vector x, namely 1
rows_rowvector = lambda x: 1

# int rows(matrix x)
# The number of rows in the matrix x
rows_matrix = lambda x: x.shape[0]

# int cols(vector x)
# The number of columns in the vector x, namely 1
cols_vector = lambda x: 1

# int cols(row_vector x)
# The number of columns in the row vector x
cols_rowvector = lambda x: x.shape[0]

# int cols(matrix x)
# The number of columns in the matrix x
cols_matrix = lambda x: x.shape[1]


## 5.6 Reductions

## 5.6.1 Log Sum of Exponents

# real log_sum_exp(vector x)
# The natural logarithm of the sum of the exponentials of the elements in x
log_sum_exp_vector = logsumexp

# real log_sum_exp(row_vector x)
# The natural logarithm of the sum of the exponentials of the elements in x
log_sum_exp_rowvector = logsumexp

# real log_sum_exp(matrix x)
# The natural logarithm of the sum of the exponentials of the elements in x
log_sum_exp_matrix = logsumexp

## 5.6.2 Minimum and Maximum

# real min(vector x)
# The minimum value in x, or +∞ if x is empty
min_vector = min

# real min(row_vector x)
# The minimum value in x, or +∞ if x is empty
min_rowvector = min

# real min(matrix x)
# The minimum value in x, or +∞ if x is empty
min_matrix = min

# real max(vector x)
# The maximum value in x, or −∞ if x is empty
max_vector = max

# real max(row_vector x)
# The maximum value in x, or −∞ if x is empty
max_rowvector = max

# real max(matrix x)
# The maximum value in x, or −∞ if x is empty
max_matrix = max

# 5.6.3 Sums and Products

# real sum(vector x)
# The sum of the values in x, or 0 if x is empty
sum_vector = sum

# real sum(row_vector x)
# The sum of the values in x, or 0 if x is empty
sum_rowvector = sum

# real sum(matrix x)
# The sum of the values in x, or 0 if x is empty
sum_matrix = sum

# real prod(vector x)
# The product of the values in x, or 1 if x is empty
prod_vector = prod

# real prod(row_vector x)
# The product of the values in x, or 1 if x is empty
prod_rowvector = prod

# real prod(matrix x)
# The product of the values in x, or 1 if x is empty
prod_matrix = prod

## 5.6.4 Sample Moments

# real mean(vector x)
# The sample mean of the values in x; see section array reductions for details.
mean_vector = mean

# real mean(row_vector x)
# The sample mean of the values in x; see section array reductions for details.
mean_rowvector = mean

# real mean(matrix x)
# The sample mean of the values in x; see section array reductions for details.
mean_matrix = mean

# real variance(vector x)
# The sample variance of the values in x; see section array reductions for details.
variance_vector = var

# real variance(row_vector x)
# The sample variance of the values in x; see section array reductions for details.
variance_rowvector = var

# real variance(matrix x)
# The sample variance of the values in x; see section array reductions for details.
variance_matrix = var

# real sd(vector x)
# The sample standard deviation of the values in x; see section array reductions for details.
sd_vector = std

# real sd(row_vector x)
# The sample standard deviation of the values in x; see section array reductions for details.
sd_rowvector = std

# real sd(matrix x)
# The sample standard deviation of the values in x; see section array reductions for details.
sd_matrix = std

## 5.7 Broadcast Functions

# vector rep_vector(real x, int m)
# Return the size m (column) vector consisting of copies of x.
rep_vector_real_int = lambda x, m: x * ones(m)
rep_vector_int_int = lambda x, m: x * ones(m)

# row_vector rep_row_vector(real x, int n)
# Return the size n row vector consisting of copies of x.
rep_row_vector_real_int = lambda x, m: x * ones(m)
rep_row_vector_int_int = lambda x, m: x * ones(m)

# matrix rep_matrix(real x, int m, int n)
# Return the m by n matrix consisting of copies of x.
rep_matrix_real_int_int = lambda x, m, n: x * ones([m, n])
rep_matrix_int_int_int = lambda x, m, n: x * ones([m, n])

# matrix rep_matrix(vector v, int n)
# Return the m by n matrix consisting of n copies of the (column) vector v of size m.
rep_matrix_vector_int = lambda v, n: v.expand([n, v.shape[0]]).t()

# matrix rep_matrix(row_vector rv, int m)
# Return the m by n matrix consisting of m copies of the row vector rv of size n.
rep_matrix_rowvector_int = lambda rv, m: rv.expand([m, rv.shape[0]])


## 5.9 Slicing and Blocking Functions

## 5.9.1 Columns and Rows

# vector col(matrix x, int n)
# The n-th column of matrix x
col_matrix_int = lambda x, n: x[:, n - 1]

# row_vector row(matrix x, int m)
# The m-th row of matrix x
row_matrix_int = lambda x, m: x[m - 1]

## 5.9.2 Block Operations

## 5.9.2.1 Matrix Slicing Operations

# matrix block(matrix x, int i, int j, int n_rows, int n_cols)
# Return the submatrix of x that starts at row i and column j and extends n_rows rows and n_cols columns.
block_matrix_int_int_int_int = lambda x, i, j, n_rows, n_cols: x[i - 1 : i - 1 + n_rows, j - 1 : j - 1 + n_cols]

# vector sub_col(matrix x, int i, int j, int n_rows)
# Return the sub-column of x that starts at row i and column j and extends n_rows rows and 1 column.
sub_col_matrix_int_int_int = lambda x, i, j, n_rows: x[i - 1 : i - 1 + n_rows, j - 1 : j]

# row_vector sub_row(matrix x, int i, int j, int n_cols)
# Return the sub-row of x that starts at row i and column j and extends 1 row and n_cols columns.
sub_row_matrix_int_int_int = lambda x, i, y, n_cols: x[i - 1 : i, j - 1 : j - 1 + n_cols]

# 5.9.2.2 Vector and Array Slicing Operations

# vector head(vector v, int n)
# Return the vector consisting of the first n elements of v.
head_vector_int = lambda v, n: v[0:n]

# row_vector head(row_vector rv, int n)
# Return the row vector consisting of the first n elements of rv.
head_rowvector_int = lambda v, n: v[0:n]

# T[] head(T[] sv, int n)
# Return the array consisting of the first n elements of sv; applies to up to three-dimensional arrays containing any type of elements T.
head_array_int = lambda v, n: v[0:n]

# vector tail(vector v, int n)
# Return the vector consisting of the last n elements of v.
tail_vector_int = lambda v, n: v[v.shape[0] - n :]

# row_vector tail(row_vector rv, int n)
# Return the row vector consisting of the last n elements of rv.
tail_rowvector_int = lambda v, n: v[v.shape[0] - n :]

# T[] tail(T[] sv, int n)
# Return the array consisting of the last n elements of sv; applies to up to three-dimensional arrays containing any type of elements T.
tail_array_int = lambda v, n: v[v.shape[0] - n :]

# vector segment(vector v, int i, int n)
# Return the vector consisting of the n elements of v starting at i; i.e., elements i through through i + n - 1.
segment_vector_int_int = lambda v, i, n: v[i - 1 : i - 1 + n]

# row_vector segment(row_vector rv, int i, int n)
# Return the row vector consisting of the n elements of rv starting at i; i.e., elements i through through i + n - 1.
segment_rowvector_int_int = lambda v, i, n: v[i - 1 : i - 1 + n]

# T[] segment(T[] sv, int i, int n)
# Return the array consisting of the n elements of sv starting at i; i.e., elements i through through i + n - 1. Applies to up to three-dimensional arrays containing any type of elements T.
segment_array_int_int = lambda v, i, n: v[i - 1 : i - 1 + n]


## 5.10 Matrix Concatenation

# Horizontal concatenation
from torch import cat

# matrix append_col(matrix x, matrix y)
# Combine matrices x and y by columns. The matrices must have the same number of rows.
append_col_matrix_matrix = lambda x, y: cat([x.t(), y.t()]).t()

# matrix append_col(matrix x, vector y)
# Combine matrix x and vector y by columns. The matrix and the vector must have the same number of rows.
append_col_matrix_vector = lambda x, y: cat([x.t(), y.expand(1,y.shape[0])]).t()

# matrix append_col(vector x, matrix y)
# Combine vector x and matrix y by columns. The vector and the matrix must have the same number of rows.
append_col_vector_matrix = lambda x, y: cat([x.expand(1,x.shape[0]), y.t()]).t()

# matrix append_col(vector x, vector y)
# Combine vectors x and y by columns. The vectors must have the same number of rows.
append_col_vector_vector = lambda x, y: cat([x.expand(1,x.shape[0]), y.expand(1,y.shape[0])]).t()

# row_vector append_col(row_vector x, row_vector y)
# Combine row vectors x and y of any size into another row vector.
append_col_rowvector_rowvector = lambda x, y: cat(x, y)

# row_vector append_col(real x, row_vector y)
# Append x to the front of y, returning another row vector.
append_col_real_rowvector = lambda x, y: cat([tensor([x], dtype=torch.float), y])
append_col_int_rowvector = lambda x, y: cat([tensor([x], dtype=torch.float), y])

# row_vector append_col(row_vector x, real y)
# Append y to the end of x, returning another row vector.
append_col_rowvector_real = lambda x, y: cat([x, tensor([y], dtype=torch.float)])
append_col_rowvector_int = lambda x, y: cat([x, tensor([y], dtype=torch.float)])

# 5.10.0.2 Vertical concatenation

# matrix append_row(matrix x, matrix y)
# Combine matrices x and y by rows. The matrices must have the same number of columns.
append_row_matrix_matrix = lambda x, y: cat([x, y])

# matrix append_row(matrix x, row_vector y)
# Combine matrix x and row vector y by rows. The matrix and the row vector must have the same number of columns.
append_row_matrix_rowvector = lambda x, y: cat([x, y.expand(1,y.shape[0])])

# matrix append_row(row_vector x, matrix y)
# Combine row vector x and matrix y by rows. The row vector and the matrix must have the same number of columns.
append_row_rowvector_matrix = lambda x, y: cat([x.expand(1,x.shape[0]), y])

# matrix append_row(row_vector x, row_vector y)
# Combine row vectors x and y by row. The row vectors must have the same number of columns.
append_row_rowvector_rowvector = lambda x, y: cat([x.expand(1,x.shape[0]), y.expand(1,y.shape[0])])

# vector append_row(vector x, vector y)
# Concatenate vectors x and y of any size into another vector.
append_row_vector_vector = lambda x, y: cat([x, y])

# vector append_row(real x, vector y)
# Append x to the top of y, returning another vector.
append_row_real_vector = lambda x, y: cat([tensor([x], dtype=torch.float), y])
append_row_int_vector = lambda x, y: cat([tensor([x], dtype=torch.float), y])

# vector append_row(vector x, real y)
# Append y to the bottom of x, returning another vector.
append_row_vector_real = lambda x, y: cat([x, tensor([y], dtype=torch.float)])
append_row_vector_int = lambda x, y: cat([x, tensor([y], dtype=torch.float)])
