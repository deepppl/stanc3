import math
import torch
import torch.nn
from torch import ones, tensor

def _XXX_TODO_XXX_(f):
    def todo(*args):
        assert False, f'{f}: not yet implemented'
    return todo

machine_precision = lambda: tensor(10 ** (-15.95), dtype=torch.float)

# 3.2 Mathematical Constants

# real pi()
# π, the ratio of a circle’s circumference to its diameter
pi = lambda: tensor(math.pi, dtype=torch.float)

# real e()
# e, the base of the natural logarithm
e = lambda: tensor(math.e, dtype=torch.float)

# real sqrt2()
# The square root of 2
from math import sqrt as msqrt
sqrt2 = lambda: tensor(msqrt(2), dtype=torch.float)

# real log2()
# The natural logarithm of 2
from math import log as mlog
log2 = lambda: tensor(mlog(2), dtype=torch.float)

# real log10()
# The natural logarithm of 10
log10 = lambda: tensor(mlog(10), dtype=torch.float)

## 3.7 Step-like Functions

## 3.7.1 Absolute Value Functions

# R fabs(T x)
# absolute value of x
from torch import abs as tabs
mabs = abs
abs_int = lambda x: array(mabs(x), dtype=dtype_float)
abs_real = tabs
abs_vector = tabs
abs_rowvector = tabs
abs_matrix = tabs
abs_array = tabs

# real fdim(real x, real y)
# Return the positive difference between x and y, which is x - y if x is greater than y and 0 otherwise; see warning above.
fdim_real_real = lambda x, y: max(x - y, 0)

## 3.7.2 Bounds Functions

# real fmin(real x, real y)
# Return the minimum of x and y; see warning above.
fmin_real_real = lambda x, y: min(x, y)
fmin_int_real = lambda x, y: min(x, y)
fmin_real_int = lambda x, y: min(x, y)
fmin_int_int = lambda x, y: min(x, y)

# real fmax(real x, real y)
# Return the maximum of x and y; see warning above.
fmax_real_real = lambda x, y: max(x, y)
fmax_int_real = lambda x, y: max(x, y)
fmax_real_int = lambda x, y: max(x, y)
fmax_int_int = lambda x, y: max(x, y)

## 3.7.3 Arithmetic Functions

# real fmod(real x, real y)
# Return the real value remainder after dividing x by y; see warning above.
fmod_real_real = lambda x, y: x % y


## 3.7.4 Rounding Functions

# R floor(T x)
# floor of x, which is the largest integer less than or equal to x, converted to a real value; see warning at start of section step-like functions
from torch import floor as tfloor
from math import floor as mfloor
floor_int = lambda x: array(mfloor(x), dtype=dtype_float)
floor_real = tfloor
floor_vector = tfloor
floor_rowvector = tfloor
floor_matrix = tfloor
floor_array = tfloor

# R ceil(T x)
# ceiling of x, which is the smallest integer greater than or equal to x, converted to a real value; see warning at start of section step-like functions
from torch import ceil as tceil
from math import ceil as mceil
ceil_int = lambda x: array(mceil(x), dtype=dtype_float)
ceil_real = tceil
ceil_vector = tceil
ceil_rowvector = tceil
ceil_matrix = tceil
ceil_array = tceil

# R round(T x)
# nearest integer to x, converted to a real value; see warning at start of section step-like functions
from torch import round as tround
mround = round
round_int = lambda x: array(mround(x), dtype=dtype_float)
round_real = tround
round_vector = tround
round_rowvector = tround
round_matrix = tround
round_array = tround

# R trunc(T x)
# integer nearest to but no larger in magnitude than x, converted to a double value; see warning at start of section step-like functions
from torch import trunc as ttrunc
from math import trunc as mtrunc
trunc_int = lambda x: array(mtrunc(x), dtype=dtype_float)
trunc_real = ttrunc
trunc_vector = ttrunc
trunc_rowvector = ttrunc
trunc_matrix = ttrunc
trunc_array = ttrunc

## 3.8 Power and Logarithm Functions

# R sqrt(T x)
# square root of x
from torch import sqrt as tsqrt
sqrt_int = lambda x: array(msqrt(x), dtype=dtype_float)
sqrt_real = tsqrt
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
square_real = tsquare
square_vector = tsquare
square_rowvector = tsquare
square_matrix = tsquare
square_array = tsquare


# R exp(T x)
# natural exponential of x
from torch import exp as texp
from math import exp as mexp
exp_int = lambda x: array(mexp(x), dtype=dtype_float)
exp_real = texp
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
log_int = lambda x: array(mlog(x), dtype=dtype_float)
log_real = tlog
log_vector = tlog
log_rowvector = tlog
log_matrix = tlog
log_array = tlog

# R log2(T x)
# base-2 logarithm of x
from torch import log2 as tlog2
from math import log2 as mlog2
log2_int = lambda x: array(mlog2(x), dtype=dtype_float)
log2_real = tlog2
log2_vector = tlog2
log2_rowvector = tlog2
log2_matrix = tlog2
log2_array = tlog2

# R log10(T x)
# base-10 logarithm of x
from torch import log10 as tlog10
from math import log10 as mlog10
log10_int = lambda x: array(mlog10(x), dtype=dtype_float)
log10_real = tlog10
log10_vector = tlog10
log10_rowvector = tlog10
log10_matrix = tlog10
log10_array = tlog10

# real pow(real x, real y)
# Return x raised to the power of y.
pow_int_int = lambda x, y: array(x ** y, dtype=dtype_float)
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
cos_int = lambda x: array(mcos(x), dtype=dtype_float)
cos_real = tcos
cos_vector = tcos
cos_rowvector = tcos
cos_matrix = tcos
cos_array = tcos

# R sin(T x)
# sine of the angle x (in radians)
from torch import sin as tsin
from math import sin as msin
sin_int = lambda x: array(msin(x), dtype=dtype_float)
sin_real = tsin
sin_vector = tsin
sin_rowvector = tsin
sin_matrix = tsin
sin_array = tsin

# R tan(T x)
# tangent of the angle x (in radians)
from torch import tan as ttan
from math import tan as mtan
tan_int = lambda x: array(mtan(x), dtype=dtype_float)
tan_real = ttan
tan_vector = ttan
tan_rowvector = ttan
tan_matrix = ttan
tan_array = ttan

# R acos(T x)
# principal arc (inverse) cosine (in radians) of x
from torch import acos as tacos
from math import acos as macos
acos_int = lambda x: array(macos(x), dtype=dtype_float)
acos_real = tacos
acos_vector = tacos
acos_rowvector = tacos
acos_matrix = tacos
acos_array = tacos

# R asin(T x)
# principal arc (inverse) sine (in radians) of x
from torch import asin as tasin
from math import asin as masin
asin_int = lambda x: array(masin(x), dtype=dtype_float)
asin_real = tasin
asin_vector = tasin
asin_rowvector = tasin
asin_matrix = tasin
asin_array = tasin

# R atan(T x)
# principal arc (inverse) tangent (in radians) of x, with values from −π
# to π
from torch import atan as tatan
from math import atan as matan
atan_int = lambda x: array(matan(x), dtype=dtype_float)
atan_real = tatan
atan_vector = tatan
atan_rowvector = tatan
atan_matrix = tatan
atan_array = tatan

# real atan2(real y, real x)
# Return the principal arc (inverse) tangent (in radians) of y divided by x
from torch import atan2 as tatan2
atan2_real_real = tatan2


## 3.10 Hyperbolic Trigonometric Functions

# R cosh(T x)
# hyperbolic cosine of x (in radians)
from torch import cosh as tcosh
from math import cosh as mcosh
cosh_int = lambda x: array(mcosh(x), dtype=dtype_float)
cosh_real = tcosh
cosh_vector = tcosh
cosh_rowvector = tcosh
cosh_matrix = tcosh
cosh_array = tcosh

# R sinh(T x)
# hyperbolic sine of x (in radians)
from torch import sinh as tsinh
from math import sinh as msinh
sinh_int = lambda x: array(msinh(x), dtype=dtype_float)
sinh_real = tsinh
sinh_vector = tsinh
sinh_rowvector = tsinh
sinh_matrix = tsinh
sinh_array = tsinh

# R tanh(T x)
# hyperbolic tangent of x (in radians)
from torch import tanh as ttanh
from math import tanh as mtanh
tanh_int = lambda x: array(mtanh(x), dtype=dtype_float)
tanh_real = ttanh
tanh_vector = ttanh
tanh_rowvector = ttanh
tanh_matrix = ttanh
tanh_array = ttanh

# R acosh(T x)
# inverse hyperbolic cosine (in radians)
from torch import acosh as tacosh
from math import acosh as macosh
acosh_int = lambda x: array(macosh(x), dtype=dtype_float)
acosh_real = tacosh
acosh_vector = tacosh
acosh_rowvector = tacosh
acosh_matrix = tacosh
acosh_array = tacosh

# R asinh(T x)
# inverse hyperbolic cosine (in radians)
from torch import asinh as tasinh
from math import asinh as masinh
asinh_int = lambda x: array(masinh(x), dtype=dtype_float)
asinh_real = tasinh
asinh_vector = tasinh
asinh_rowvector = tasinh
asinh_matrix = tasinh
asinh_array = tasinh

# R atanh(T x)
# inverse hyperbolic tangent (in radians) of x
from torch import atanh as tatanh
from math import atanh as matanh
atanh_int = lambda x: array(matanh(x), dtype=dtype_float)
atanh_real = tatanh
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
inv_logit_int = lambda x: sigmoid(tensor(x, dtype=torch.float))
inv_logit_real = sigmoid
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
log_inv_logit_int = lambda x: LogSigmoid()(tensor(x, dtype=torch.float))
log_inv_logit_real = LogSigmoid()
log_inv_logit_vector = LogSigmoid()
log_inv_logit_rowvector = LogSigmoid()
log_inv_logit_matrix = LogSigmoid()
log_inv_logit_array = LogSigmoid()

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
from torch import min as tmin
min_array = tmin

# real max(real[] x)
# The maximum value in x, or −∞ if x is size 0.
# int max(int[] x)
# The maximum value in x, or error if x is size 0.
from torch import max as tmax
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
log_sum_exp_array = lambda x: logsumexp(x, 0)

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
dims_int = lambda x: tensor([], dtype=torch.long)
dims_real = lambda x: tensor(x.shape)
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
num_elements_array = lambda x: x.shape.numel()

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

## 5.5 Dot Products and Specialized Products

# real dot_product(vector x, vector y)
# The dot product of x and y
from torch import dot as tdot
dot_product_vector_vector = tdot

# real dot_product(vector x, row_vector y)
# The dot product of x and y
dot_product_vector_rowvector = tdot

# real dot_product(row_vector x, vector y)
# The dot product of x and y
dot_product_rowvector_vector = tdot

# real dot_product(row_vector x, row_vector y)
# The dot product of x and y
dot_product_rowvector_rowvector = tdot

# row_vector columns_dot_product(vector x, vector y)
# The dot product of the columns of x and y
columns_dot_product_vector_vector = _XXX_TODO_XXX_('columns_dot_product')

# row_vector columns_dot_product(row_vector x, row_vector y)
# The dot product of the columns of x and y
columns_dot_product_rowvector_rowvector = _XXX_TODO_XXX_('columns_dot_product')

# row_vector columns_dot_product(matrix x, matrix y)
# The dot product of the columns of x and y
columns_dot_product_matrix_matrix = _XXX_TODO_XXX_('columns_dot_product')

# vector rows_dot_product(vector x, vector y)
# The dot product of the rows of x and y
rows_dot_product_vector_vector = _XXX_TODO_XXX_('rows_dot_product')

# vector rows_dot_product(row_vector x, row_vector y)
# The dot product of the rows of x and y
rows_dot_product_rowvector_rowvector = _XXX_TODO_XXX_('rows_dot_product')

# vector rows_dot_product(matrix x, matrix y)
# The dot product of the rows of x and y
rows_dot_product_matrix_matrix = _XXX_TODO_XXX_('rows_dot_product')

# real dot_self(vector x)
# The dot product of the vector x with itself
dot_self_vector = lambda x: dot_product_vector_vector(x, x)

# real dot_self(row_vector x)
# The dot product of the row vector x with itself
dot_self_rowvector = lambda x: dot_product_rowvector_rowvector(x, x)

# row_vector columns_dot_self(vector x)
# The dot product of the columns of x with themselves
columns_dot_self_vector = lambda x: columns_dot_product_vector_vector(x, x)

# row_vector columns_dot_self(row_vector x)
# The dot product of the columns of x with themselves
columns_dot_self_rowvector = lambda x: columns_dot_product_rowvector_rowvector(x, x)

# row_vector columns_dot_self(matrix x)
# The dot product of the columns of x with themselves
columns_dot_self_matrix = lambda x: columns_dot_product_matrix_matrix(x, x)

# vector rows_dot_self(vector x)
# The dot product of the rows of x with themselves
rows_dot_self_vector = lambda x: rows_dot_product_vector_vector(x, x)

# vector rows_dot_self(row_vector x)
# The dot product of the rows of x with themselves
rows_dot_self_rowvector = lambda x: rows_dot_product_rowvector_rowvector(x, x)

# vector rows_dot_self(matrix x)
# The dot product of the rows of x with themselves
rows_dot_self_matrix = lambda x: rows_dot_product_matrix_matrix(x, x)

## 5.5.1 Specialized Products

# matrix tcrossprod(matrix x)
# The product of x postmultiplied by its own transpose, similar to the tcrossprod(x) function in R. The result is a symmetric matrix.
tcrossprod_matrix = _XXX_TODO_XXX_('tcrossprod')

# matrix crossprod(matrix x)
# The product of x premultiplied by its own transpose, similar to the crossprod(x) function in R. The result is a symmetric matrix.
crossprod_matrix = _XXX_TODO_XXX_('crossprod')

# matrix quad_form(matrix A, matrix B)
# The quadratic form, i.e., B' * A * B.
quad_form_matrix_matrix = _XXX_TODO_XXX_('quad_form')

# real quad_form(matrix A, vector B)
# The quadratic form, i.e., B' * A * B.
quad_form_matrix_vector = _XXX_TODO_XXX_('quad_form')

# matrix quad_form_diag(matrix m, vector v)
# The quadratic form using the column vector v as a diagonal matrix, i.e., diag_matrix(v) * m * diag_matrix(v).
quad_form_diag_matrix_vector = _XXX_TODO_XXX_('quad_form_diag')

# matrix quad_form_diag(matrix m, row_vector rv)
# The quadratic form using the row vector rv as a diagonal matrix, i.e., diag_matrix(rv) * m * diag_matrix(rv).
quad_form_diag_matrix_row_vector  = _XXX_TODO_XXX_('quad_form_diag')

# matrix quad_form_sym(matrix A, matrix B)
# Similarly to quad_form, gives B' * A * B, but additionally checks if A is symmetric and ensures that the result is also symmetric.
quad_form_sym_matrix_matrix = _XXX_TODO_XXX_('quad_form_sym')

# real quad_form_sym(matrix A, vector B)
# Similarly to quad_form, gives B' * A * B, but additionally checks if A is symmetric and ensures that the result is also symmetric.
quad_form_sym_matrix_vector = _XXX_TODO_XXX_('quad_form_sym')

# real trace_quad_form(matrix A, matrix B)
# The trace of the quadratic form, i.e., trace(B' * A * B).
trace_quad_form_matrix_matrix = _XXX_TODO_XXX_('trace_quad_form')

# real trace_gen_quad_form(matrix D, matrix A, matrix B)
# The trace of a generalized quadratic form, i.e., trace(D * B' * A * B).
trace_gen_quad_form_matrix_matrix_matrix = _XXX_TODO_XXX_('trace_gen_quad_form')

# matrix multiply_lower_tri_self_transpose(matrix x)
# The product of the lower triangular portion of x (including the diagonal) times its own transpose; that is, if L is a matrix of the same dimensions as x with L(m,n) equal to x(m,n) for n≤m
# and L(m,n) equal to 0 if n>m, the result is the symmetric matrix LL⊤. This is a specialization of tcrossprod(x) for lower-triangular matrices. The input matrix does not need to be square.
multiply_lower_tri_self_matrix = _XXX_TODO_XXX_('multiply_lower_tri_self')

# matrix diag_pre_multiply(vector v, matrix m)
# Return the product of the diagonal matrix formed from the vector v and the matrix m, i.e., diag_matrix(v) * m.
diag_pre_multiply_vector_matrix = _XXX_TODO_XXX_('diag_pre_multiply')

# matrix diag_pre_multiply(row_vector rv, matrix m)
# Return the product of the diagonal matrix formed from the vector rv and the matrix m, i.e., diag_matrix(rv) * m.
diag_pre_multiply_rowvector_matrix = _XXX_TODO_XXX_('diag_pre_multiply')

# matrix diag_post_multiply(matrix m, vector v)
# Return the product of the matrix m and the diagonal matrix formed from the vector v, i.e., m * diag_matrix(v).
diag_post_multiply_matrix_vector = _XXX_TODO_XXX_('diag_post_multiply')

# matrix diag_post_multiply(matrix m, row_vector rv)
# Return the product of the matrix m and the diagonal matrix formed from the the row vector rv, i.e., m * diag_matrix(rv).
diag_post_multiply_matrix_rowvector = _XXX_TODO_XXX_('diag_post_multiply')

## 5.6 Reductions

## 5.6.1 Log Sum of Exponents

# real log_sum_exp(vector x)
# The natural logarithm of the sum of the exponentials of the elements in x
log_sum_exp_vector = lambda x: logsumexp(x, 0)

# real log_sum_exp(row_vector x)
# The natural logarithm of the sum of the exponentials of the elements in x
log_sum_exp_rowvector = lambda x: logsumexp(x, 0)

# real log_sum_exp(matrix x)
# The natural logarithm of the sum of the exponentials of the elements in x
log_sum_exp_matrix = lambda x: logsumexp(x, (0, 1))

## 5.6.2 Minimum and Maximum

# real min(vector x)
# The minimum value in x, or +∞ if x is empty
min_vector = tmin

# real min(row_vector x)
# The minimum value in x, or +∞ if x is empty
min_rowvector = tmin

# real min(matrix x)
# The minimum value in x, or +∞ if x is empty
min_matrix = tmin

# real max(vector x)
# The maximum value in x, or −∞ if x is empty
max_vector = tmax

# real max(row_vector x)
# The maximum value in x, or −∞ if x is empty
max_rowvector = tmax

# real max(matrix x)
# The maximum value in x, or −∞ if x is empty
max_matrix = tmax

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

## 5.8 Diagonal Matrix Functions

from torch import diag, eye

# matrix add_diag(matrix m, row_vector d)
# Add row_vector d to the diagonal of matrix m.

add_diag_matrix_rowvector = lambda m, rv: m + diag(rv)

# matrix add_diag(matrix m, vector d)
# Add vector d to the diagonal of matrix m.

add_diag_matrix_vector = lambda m, v: m + diag(v)

# matrix add_diag(matrix m, real d)
# Add scalar d to every diagonal element of matrix m.

add_diag_matrix_real = lambda m, r: r*eye(m.shape[0]) + m

# vector diagonal(matrix x)
# The diagonal of the matrix x

diagonal_matrix = lambda m: diag(m)

# matrix diag_matrix(vector x)
# The diagonal matrix with diagonal x

diag_matrix_vector = lambda v: diag(v)

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
sub_col_matrix_int_int_int = lambda x, i, j, n_rows: (x[i - 1 : i - 1 + n_rows, j - 1 : j])[:,0]

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

## 5.10.0.1 Horizontal concatenation
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

## 5.10.0.2 Vertical concatenation

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

## 5.11 Special Matrix Functions
## 5.11.1 Softmax

# vector softmax(vector x)
# The softmax of x
from torch.nn import Softmax as tSoftmax
softmax_vector = lambda x: tSoftmax(dim=x.shape)(x)

# vector log_softmax(vector x)
# The natural logarithm of the softmax of x
from torch.nn import LogSoftmax as tLogSoftmax
softmax_vector = lambda x: tLogSoftmax(dim=x.shape)(x)

# 5.11.2 Cumulative Sums

# real[] cumulative_sum(real[] x)
# The cumulative sum of x
from torch import cumsum as tcumsum
cumulative_sum_array = lambda x: tcumsum(x, dim=0)

# vector cumulative_sum(vector v)
# The cumulative sum of v
cumulative_sum_vector = lambda x: tcumsum(x, dim=0)

# row_vector cumulative_sum(row_vector rv)
# The cumulative sum of rv
cumulative_sum_rowvector = lambda x: tcumsum(x, dim=0)

## 5.12 Covariance Functions
## 5.12.1 Exponentiated quadratic covariance function

def cov_exp_quad(x, alpha, rho):
    return alpha * alpha * texp(-0.5 * torch.pow(torch.cdist(x, x) / rho, 2))

# matrix cov_exp_quad(row_vectors x, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x.
cov_exp_quad_rowvector_real_real = cov_exp_quad

# matrix cov_exp_quad(vectors x, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x.
cov_exp_quad_vector_real_real = cov_exp_quad

# matrix cov_exp_quad(real[] x, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x.
cov_exp_quad_array_real_real = lambda x, alpha, rho: cov_exp_quad(x.view(1, -1), alpha, rho)

# matrix cov_exp_quad(row_vectors x1, row_vectors x2, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x1 and x2.
cov_exp_quad_rowvector_rowvector_real_real = cov_exp_quad

# matrix cov_exp_quad(vectors x1, vectors x2, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x1 and x2.
cov_exp_quad_vector_vector_real_real = cov_exp_quad

# matrix cov_exp_quad(real[] x1, real[] x2, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x1 and x2.
cov_exp_quad_array_array_real_real = cov_exp_quad

# 5.13 Linear Algebra Functions and Solvers

# 5.13.4.6 Cholesky Decomposition

from torch import cholesky

# matrix cholesky_decompose(matrix A)
# The lower-triangular Cholesky factor of the symmetric positive-definite matrix A

cholesky_decompose_matrix = lambda m: cholesky(m)

## 7. Mixed Operations

# matrix to_matrix(matrix m)
# Return the matrix m itself.
to_matrix_matrix = lambda m: m

# matrix to_matrix(vector v)
# Convert the column vector v to a size(v) by 1 matrix.
to_matrix_vector = lambda v: v.expand(1, v.shape[0])

# matrix to_matrix(row_vector v)
# Convert the row vector v to a 1 by size(v) matrix.
to_matrix_rowvector = lambda v: v.expand(1, v.shape[0]).t()

# matrix to_matrix(matrix m, int m, int n)
# Convert a matrix m to a matrix with m rows and n columns filled in column-major order.
to_matrix_matrix_int_int = lambda mat, m, n: mat.t().reshape(m,n)

# matrix to_matrix(vector v, int m, int n)
# Convert a vector v to a matrix with m rows and n columns filled in column-major order.
to_matrix_vector_int_int = _XXX_TODO_XXX_('to_matrix_vector_int_int')

# matrix to_matrix(row_vector v, int m, int n)
# Convert a row_vector a to a matrix with m rows and n columns filled in column-major order.
to_matrix_rowvector_int_int = _XXX_TODO_XXX_('to_matrix')

# matrix to_matrix(matrix m, int m, int n, int col_major)
# Convert a matrix m to a matrix with m rows and n columns filled in row-major order if col_major equals 0 (otherwise, they get filled in column-major order).
def to_matrix_matrix_int_int_int(mat, m, n, col_major):
    if col_major == 0:
        mat.reshape(m,n)
    else:
        to_matrix_matrix_int_int(mat, m, n)

# matrix to_matrix(vector v, int m, int n, int col_major)
# Convert a vector v to a matrix with m rows and n columns filled in row-major order if col_major equals 0 (otherwise, they get filled in column-major order).
def to_matrix_vector_int_int_int(v, m, n, col_major):
    if col_major == 0:
        v.reshape(m,n)
    else:
        to_matrix_vector_int_int(v, m, n)

# matrix to_matrix(row_vector v, int m, int n, int col_major)
# Convert a row_vector a to a matrix with m rows and n columns filled in row-major order if col_major equals 0 (otherwise, they get filled in column-major order).
def to_matrix_rowvector_int_int_int(v, m, n, col_major):
    if col_major == 0:
        v.reshape(m,n)
    else:
        to_matrix_rowvector_int_int(v, m, n)

# matrix to_matrix(real[] a, int m, int n)
# Convert a one-dimensional array a to a matrix with m rows and n columns filled in column-major order.
# matrix to_matrix(int[] a, int m, int n)
# Convert a one-dimensional array a to a matrix with m rows and n columns filled in column-major order.
to_matrix_array_int_int = to_matrix_vector_int_int

# matrix to_matrix(real[] a, int m, int n, int col_major)
# Convert a one-dimensional array a to a matrix with m rows and n columns filled in row-major order if col_major equals 0 (otherwise, they get filled in column-major order).
# matrix to_matrix(int[] a, int m, int n, int col_major)
# Convert a one-dimensional array a to a matrix with m rows and n columns filled in row-major order if col_major equals 0 (otherwise, they get filled in column-major order).
to_matrix_array_int_int_int = to_matrix_vector_int_int_int

# matrix to_matrix(real[,] a)
# Convert the two dimensional array a to a matrix with the same dimensions and indexing order.
# matrix to_matrix(int[,] a)
# Convert the two dimensional array a to a matrix with the same dimensions and indexing order. If any of the dimensions of a are zero, the result will be a 0×0 matrix.
to_matrix_array = lambda a: a

# vector to_vector(matrix m)
# Convert the matrix m to a column vector in column-major order.
to_vector_matrix = lambda m: m.reshape(-1)

# vector to_vector(vector v)
# Return the column vector v itself.
to_vector_vector = lambda v: v

# vector to_vector(row_vector v)
# Convert the row vector v to a column vector.
to_vector_rowvector = lambda v: v

# vector to_vector(real[] a)
# Convert the one-dimensional array a to a column vector.
# vector to_vector(int[] a)
# Convert the one-dimensional integer array a to a column vector.
to_vector_array = lambda v: v

# row_vector to_row_vector(matrix m)
# Convert the matrix m to a row vector in column-major order.
to_row_vector_matrix = lambda m: m.reshape(-1)

# row_vector to_row_vector(vector v)
# Convert the column vector v to a row vector.
to_row_vector_vector = lambda v: v

# row_vector to_row_vector(row_vector v)
# Return the row vector v itself.
to_row_vector_rowvector = lambda v: v

# row_vector to_row_vector(real[] a)
# Convert the one-dimensional array a to a row vector.
# row_vector to_row_vector(int[] a)
# Convert the one-dimensional array a to a row vector.
to_row_vector_array = lambda v: v

# real[,] to_array_2d(matrix m)
# Convert the matrix m to a two dimensional array with the same dimensions and indexing order.
to_array_2d_matrix = lambda m: m

# real[] to_array_1d(vector v)
# Convert the column vector v to a one-dimensional array.
to_array_1d_vector = lambda v: v

# real[] to_array_1d(row_vector v)
# Convert the row vector v to a one-dimensional array.
to_array_1d_rowvector = lambda v: v

# real[] to_array_1d(matrix m)
# Convert the matrix m to a one-dimensional array in column-major order.
to_array_1d_matrix = lambda m: m.t().reshape(-1)

# real[] to_array_1d(real[...] a)
# Convert the array a (of any dimension up to 10) to a one-dimensional array in row-major order.
# int[] to_array_1d(int[...] a)
# Convert the array a (of any dimension up to 10) to a one-dimensional array in row-major order.
to_array_1d_array = lambda a: a.reshape(-1)

# 9.2 Ordinary Differential Equation (ODE) Solvers

# 9.2.2 Non-Stiff Solver

# real[ , ] integrate_ode_rk45(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, real[] x_r, int[] x_i)
# Solves the ODE system for the times provided using the Runge Kutta Dopri algorithm with the implementation from Boost.

from torchdiffeq import odeint

def integrate_ode_rk45_array_real_array_array_array_array(ode, initial_state, initial_time, times, theta, x_r, x_i):
    f = lambda t, y: ode(t, y, theta, x_r, x_i) 
    odeint(f, initial_state, times)

# real[ , ] integrate_ode_rk45(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, real[] x_r, int[] x_i, real rel_tol, real abs_tol, int max_num_steps)
# Solves the ODE system for the times provided using the Runge Kutta Dopri algorithm with the implementation from Boost with additional control parameters for the solver.

def integrate_ode_rk45_array_real_array_array_array_array_real_real_int(ode, initial_state, times, theta, x_r, x_i, rtol, atol, mxstep):
    f = lambda t, y: ode(t, y, theta, x_r, x_i) 
    odeint(f, initial_state, times, rtol=rtol, atol=atol, options={"max_num_steps": mxstep})