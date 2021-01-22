import math
import jax
import jax.numpy as tensor
from jax.numpy import ones, array

dtype_float = tensor.dtype("float32")
dtype_long = tensor.dtype("int32")

## Utility functions

def _XXX_TODO_XXX_(f):
    def todo(*args):
        assert False, f"{f}: not yet implemented"

    return todo

def _to_float_1(f):
    def f_casted(y, *args):
        return f(array(x, dtype=dtype_float), *args)

    return f_casted

def _to_float_2(f):
    def f_casted(x, y, *args):
        return f(x, array(y, dtype=dtype_float), *args)

    return f_casted

def _to_float_1_2(f):
    def f_casted(x, y, *args):
        return f(array(x, dtype=dtype_float), array(y, dtype=dtype_float), *args)

    return f_casted


## Based on Stan Functions Reference
# version 2.26 (https://mc-stan.org/docs/2_26/functions-reference/index.html)

## 1 Void Functions

## 1.1 Print statement

# void print(T1 x1,..., TN xN)
# This function is directly supported by the compiler

## 1.2 Reject statement

# void reject(T1 x1,..., TN xN)
# This function is directly supported by the compiler


## 2 Integer-Valued Basic Functions

## 2.1 Integer-valued arithmetic operators

## 2.1.1 Binary infix operators

# int operator+(int x, int y)
# The sum of the addends x and y
# This function is directly supported by the compiler

# int operator-(int x, int y)
# The difference between the minuend x and subtrahend y
# This function is directly supported by the compiler

# int operator*(int x, int y)
# The product of the factors x and y
# This function is directly supported by the compiler

# int operator/(int x, int y)
# The integer quotient of the dividend x and divisor y
# This function is directly supported by the compiler

# int operator%(int x, int y)
# x modulo y, which is the positive remainder after dividing x by y. If both x and y are non-negative, so is the result; otherwise, the sign of the result is platform dependent.
# This function is directly supported by the compiler

## 2.1.2 Unary prefix operators

# int operator-(int x)
# The negation of the subtrahend x
# This function is directly supported by the compiler

# int operator+(int x)
# This is a no-op.
# This function is directly supported by the compiler

## 2.2 Absolute functions

# R abs(T x)
# absolute value of x
abs_int = abs

# int int_step(int x)
# int int_step(real x)
# Return the step function of x as an integer, int_step(x)={1if x>00if x≤0 or x is NaN
# Warning: int_step(0) and int_step(NaN) return 0 whereas step(0) and step(NaN) return 1.
int_step_int = _XXX_TODO_XXX_("int_step")
int_step_real = _XXX_TODO_XXX_("int_step")

## 2.3 Bound functions

# int min(int x, int y)
# Return the minimum of x and y.
min_int_int = min

# int max(int x, int y)
# Return the maximum of x and y.
max_int_int = max


## 2.4 Size functions

# int size(int x)
# int size(real x)
# Return the size of x which for scalar-valued x is 1
size_int = lambda x: 1
size_real = lambda x: 1


## 3 Real-Valued Basic Functions

## 3.1 Vectorization of real-valued functions

# This section does not define functions


## 3.2 Mathematical Constants

# real pi()
# π, the ratio of a circle’s circumference to its diameter
pi = lambda: array(math.pi, dtype=dtype_float)

# real e()
# e, the base of the natural logarithm
e = lambda: array(math.e, dtype=dtype_float)

# real sqrt2()
# The square root of 2
from math import sqrt as msqrt

sqrt2 = lambda: array(msqrt(2), dtype=dtype_float)

# real log2()
# The natural logarithm of 2
from math import log as mlog

log2 = lambda: array(mlog(2), dtype=dtype_float)

# real log10()
# The natural logarithm of 10
log10 = lambda: array(mlog(10), dtype=dtype_float)


## 3.3 Special Values

# real not_a_number()
# Not-a-number, a special non-finite real value returned to signal an error
not_a_number = lambda: array(math.nan, dtype=dtype_float)

# real positive_infinity()
# Positive infinity, a special non-finite real value larger than all finite numbers
positive_infinity = lambda: array(float("inf"), dtype=dtype_float)

# real negative_infinity()
# Negative infinity, a special non-finite real value smaller than all finite numbers
negative_infinity = lambda: array(float("-inf"), dtype=dtype_float)

# real machine_precision()
# The smallest number x
# such that (x+1)≠1 in floating-point arithmetic on the current hardware platform
machine_precision = lambda: array(10 ** (-15.95), dtype=dtype_float)


## 3.4 Log probability function

# real target()
# Return the current value of the log probability accumulator.
target = _XXX_TODO_XXX_("target")

# real get_lp()
# Return the current value of the log probability accumulator; deprecated; - use target() instead.
get_lp = _XXX_TODO_XXX_("get_lp")


## 3.5 Logical functions

## 3.5.1 Comparison operators

# int operator<(int x, int y)
# int operator<(real x, real y)
# Return 1 if x is less than y and 0 otherwise.
# This function is directly supported by the compiler

# int operator<=(int x, int y)
# int operator<=(real x, real y)
# Return 1 if x is less than or equal y and 0 otherwise.
# This function is directly supported by the compiler

# int operator>(int x, int y)
# int operator>(real x, real y)
# Return 1 if x is greater than y and 0 otherwise.
# This function is directly supported by the compiler

# int operator>=(int x, int y)
# int operator>=(real x, real y)
# Return 1 if x is greater than or equal to y and 0 otherwise.
# This function is directly supported by the compiler

# int operator==(int x, int y)
# int operator==(real x, real y)
# Return 1 if x is equal to y and 0 otherwise.
# This function is directly supported by the compiler

# int operator!=(int x, int y)
# int operator!=(real x, real y)
# Return 1 if x is not equal to y and 0 otherwise.
# This function is directly supported by the compiler


## 3.5.2 Boolean operators

# int operator!(int x)
# int operator!(real x)
# Return 1 if x is zero and 0 otherwise.
# This function is directly supported by the compiler

# int operator&&(int x, int y)
# int operator&&(real x, real y)
# Return 1 if x is unequal to 0 and y is unequal to 0.
# This function is directly supported by the compiler


# int operator||(int x, int y)
# int operator||(real x, real y)
# Return 1 if x is unequal to 0 or y is unequal to 0.
# This function is directly supported by the compiler


## 3.5.3 Logical functions

# real step(real x)
# Return 1 if x is positive and 0 otherwise.
# Warning: int_step(0) and int_step(NaN) return 0 whereas step(0) and step(NaN) return 1.
step_real = _XXX_TODO_XXX_("step")

# int is_inf(real x)
# Return 1 if x is infinite (positive or negative) and 0 otherwise.
is_inf = _XXX_TODO_XXX_("is_inf")

# int is_nan(real x)
# Return 1 if x is NaN and 0 otherwise.
is_nan = _XXX_TODO_XXX_("is_nan")


## 3.6 Real-valued arithmetic operators

## 3.6.1 Binary infix operators

# real operator+(real x, real y)
# Return the sum of x and y.
# This function is directly supported by the compiler

# real operator-(real x, real y)
# Return the difference between x and y.
# This function is directly supported by the compiler

# real operator*(real x, real y)
# Return the product of x and y.
# This function is directly supported by the compiler

# real operator/(real x, real y)
# Return the quotient of x and y.
# This function is directly supported by the compiler

# real operator^(real x, real y)
# Return x raised to the power of y.
# This function is directly supported by the compiler

## 3.6.2 Unary prefix operators

# real operator-(real x)
# Return the negation of the subtrahend x.
# This function is directly supported by the compiler

# real operator+(real x)
# Return the value of x.
# This function is directly supported by the compiler


## 3.7 Step-like Functions

## 3.7.1 Absolute Value Functions

# R fabs(T x)
# absolute value of x
from jax.numpy import abs as tabs

mabs = abs
abs_int = mabs
abs_real = tabs
abs_vector = tabs
abs_rowvector = tabs
abs_matrix = tabs
abs_array = tabs

# real fdim(real x, real y)
# Return the positive difference between x and y, which is x - y if x is greater than y and 0 otherwise; see warning above.
fdim_real_real = lambda x, y: max(x - y, 0)
fdim_int_real = lambda x, y: max(x - y, 0)
fdim_real_int = lambda x, y: max(x - y, 0)
fdim_int_int = lambda x, y: max(x - y, 0)

# R fdim(T1 x, T2 y)
# Vectorized implementation of the fdim function
fdim_vectorized = _XXX_TODO_XXX_("fdim")
fdim_vector_vector = fdim_vectorized
fdim_rowvector_rowvector = fdim_vectorized
fdim_matrix_matrix = fdim_vectorized
fdim_array_array = fdim_vectorized

fdim_real_vector = fdim_vectorized
fdim_real_rowvector = fdim_vectorized
fdim_real_matrix = fdim_vectorized
fdim_real_array = fdim_vectorized
fdim_vector_real = fdim_vectorized
fdim_rowvector_real = fdim_vectorized
fdim_matrix_real = fdim_vectorized
fdim_array_real = fdim_vectorized

fdim_int_vector = _to_float_1(fdim_vectorized)
fdim_int_rowvector = _to_float_1(fdim_vectorized)
fdim_int_matrix = _to_float_1(fdim_vectorized)
fdim_int_array = _to_float_1(fdim_vectorized)
fdim_vector_int = _to_float_2(fdim_vectorized)
fdim_rowvector_int = _to_float_2(fdim_vectorized)
fdim_matrix_int = _to_float_2(fdim_vectorized)
fdim_array_int = _to_float_2(fdim_vectorized)

## 3.7.2 Bounds Functions

# real fmin(real x, real y)
# Return the minimum of x and y; see warning above.
fmin_real_real = lambda x, y: min(x, y)
fmin_int_real = lambda x, y: min(x, y)
fmin_real_int = lambda x, y: min(x, y)
fmin_int_int = lambda x, y: min(x, y)

# R fmin(T1 x, T2 y)
# Vectorized implementation of the fmin function
fmin_vectorized = _XXX_TODO_XXX_("fmin")
fmin_vector_vector = fmin_vectorized
fmin_rowvector_rowvector = fmin_vectorized
fmin_matrix_matrix = fmin_vectorized
fmin_array_array = fmin_vectorized

fmin_real_vector = fmin_vectorized
fmin_real_rowvector = fmin_vectorized
fmin_real_matrix = fmin_vectorized
fmin_real_array = fmin_vectorized
fmin_vector_real = fmin_vectorized
fmin_rowvector_real = fmin_vectorized
fmin_matrix_real = fmin_vectorized
fmin_array_real = fmin_vectorized

fmin_int_vector = _to_float_1(fmin_vectorized)
fmin_int_rowvector = _to_float_1(fmin_vectorized)
fmin_int_matrix = _to_float_1(fmin_vectorized)
fmin_int_array = _to_float_1(fmin_vectorized)
fmin_vector_int = _to_float_2(fmin_vectorized)
fmin_rowvector_int = _to_float_2(fmin_vectorized)
fmin_matrix_int = _to_float_2(fmin_vectorized)
fmin_array_int = _to_float_2(fmin_vectorized)


# real fmax(real x, real y)
# Return the maximum of x and y; see warning above.
fmax_real_real = lambda x, y: max(x, y)
fmax_int_real = lambda x, y: max(x, y)
fmax_real_int = lambda x, y: max(x, y)
fmax_int_int = lambda x, y: max(x, y)

# R fmax(T1 x, T2 y)
# Vectorized implementation of the fmax function
fmax_vectorized = _XXX_TODO_XXX_("fmax")
fmax_vector_vector = fmax_vectorized
fmax_rowvector_rowvector = fmax_vectorized
fmax_matrix_matrix = fmax_vectorized
fmax_array_array = fmax_vectorized

fmax_real_vector = fmax_vectorized
fmax_real_rowvector = fmax_vectorized
fmax_real_matrix = fmax_vectorized
fmax_real_array = fmax_vectorized
fmax_vector_real = fmax_vectorized
fmax_rowvector_real = fmax_vectorized
fmax_matrix_real = fmax_vectorized
fmax_array_real = fmax_vectorized

fmax_int_vector = _to_float_1(fmax_vectorized)
fmax_int_rowvector = _to_float_1(fmax_vectorized)
fmax_int_matrix = _to_float_1(fmax_vectorized)
fmax_int_array = _to_float_1(fmax_vectorized)
fmax_vector_int = _to_float_2(fmax_vectorized)
fmax_rowvector_int = _to_float_2(fmax_vectorized)
fmax_matrix_int = _to_float_2(fmax_vectorized)
fmax_array_int = _to_float_2(fmax_vectorized)


## 3.7.3 Arithmetic Functions

# real fmod(real x, real y)
# Return the real value remainder after dividing x by y; see warning above.
fmod_real_real = lambda x, y: x % y
fmod_int_real = lambda x, y: x % y
fmod_real_int = lambda x, y: x % y
fmod_int_int = lambda x, y: x % y

# R fmod(T1 x, T2 y)
# Vectorized implementation of the fmod function
fmod_vectorized = _XXX_TODO_XXX_("fmod")
fmod_vector_vector = fmod_vectorized
fmod_rowvector_rowvector = fmod_vectorized
fmod_matrix_matrix = fmod_vectorized
fmod_array_array = fmod_vectorized

fmod_real_vector = fmod_vectorized
fmod_real_rowvector = fmod_vectorized
fmod_real_matrix = fmod_vectorized
fmod_real_array = fmod_vectorized
fmod_vector_real = fmod_vectorized
fmod_rowvector_real = fmod_vectorized
fmod_matrix_real = fmod_vectorized
fmod_array_real = fmod_vectorized

fmod_int_vector = _to_float_1(fmod_vectorized)
fmod_int_rowvector = _to_float_1(fmod_vectorized)
fmod_int_matrix = _to_float_1(fmod_vectorized)
fmod_int_array = _to_float_1(fmod_vectorized)
fmod_vector_int = _to_float_2(fmod_vectorized)
fmod_rowvector_int = _to_float_2(fmod_vectorized)
fmod_matrix_int = _to_float_2(fmod_vectorized)
fmod_array_int = _to_float_2(fmod_vectorized)


## 3.7.4 Rounding Functions

# R floor(T x)
# floor of x, which is the largest integer less than or equal to x, converted to a real value; see warning at start of section step-like functions
from math import floor as mfloor
from jax.numpy import floor as tfloor

floor_int = mfloor
floor_real = tfloor
floor_vector = tfloor
floor_rowvector = tfloor
floor_matrix = tfloor
floor_array = tfloor

# R ceil(T x)
# ceiling of x, which is the smallest integer greater than or equal to x, converted to a real value; see warning at start of section step-like functions
from math import ceil as mceil
from jax.numpy import ceil as tceil

ceil_int = mceil
ceil_real = tceil
ceil_vector = tceil
ceil_rowvector = tceil
ceil_matrix = tceil
ceil_array = tceil

# R round(T x)
# nearest integer to x, converted to a real value; see warning at start of section step-like functions
from jax.numpy import round as tround

mround = round
round_int = mround
round_real = tround
round_vector = tround
round_rowvector = tround
round_matrix = tround
round_array = tround

# R trunc(T x)
# integer nearest to but no larger in magnitude than x, converted to a double value; see warning at start of section step-like functions
from math import trunc as mtrunc
from jax.numpy import trunc as ttrunc

trunc_int = mtrunc
trunc_real = ttrunc
trunc_vector = ttrunc
trunc_rowvector = ttrunc
trunc_matrix = ttrunc
trunc_array = ttrunc

## 3.8 Power and Logarithm Functions

# R sqrt(T x)
# square root of x
from jax.numpy import sqrt as tsqrt

sqrt_int = lambda x: array(msqrt(x), dtype=dtype_float)
sqrt_real = tsqrt
sqrt_vector = tsqrt
sqrt_rowvector = tsqrt
sqrt_matrix = tsqrt
sqrt_array = tsqrt

# R cbrt(T x)
# cube root of x
cbrt_int = _XXX_TODO_XXX_("cbrt")
cbrt_real = _XXX_TODO_XXX_("cbrt")
cbrt_vector = _XXX_TODO_XXX_("cbrt")
cbrt_rowvector = _XXX_TODO_XXX_("cbrt")
cbrt_matrix = _XXX_TODO_XXX_("cbrt")
cbrt_array = _XXX_TODO_XXX_("cbrt")

# R square(T x)
# square of x
from jax.numpy import square as tsquare

square_int = lambda x: x * x
square_real = tsquare
square_vector = tsquare
square_rowvector = tsquare
square_matrix = tsquare
square_array = tsquare

# R exp(T x)
# natural exponential of x
from math import exp as mexp
from jax.numpy import exp as texp

exp_int = lambda x: array(mexp(x), dtype=dtype_float)
exp_real = texp
exp_vector = texp
exp_rowvector = texp
exp_matrix = texp
exp_array = texp

# R exp2(T x)
# base-2 exponential of x
exp2_int = _XXX_TODO_XXX_("exp2")
exp2_real = _XXX_TODO_XXX_("exp2")
exp2_vector = _XXX_TODO_XXX_("exp2")
exp2_rowvector = _XXX_TODO_XXX_("exp2")
exp2_matrix = _XXX_TODO_XXX_("exp2")
exp2_array = _XXX_TODO_XXX_("exp2")

# R log(T x)
# natural logarithm of x
from jax.numpy import log as tlog

log_int = lambda x: array(mlog(x), dtype=dtype_float)
log_real = tlog
log_vector = tlog
log_rowvector = tlog
log_matrix = tlog
log_array = tlog

# R log2(T x)
# base-2 logarithm of x
from math import log2 as mlog2
from jax.numpy import log2 as tlog2

log2_int = lambda x: array(mlog2(x), dtype=dtype_float)
log2_real = tlog2
log2_vector = tlog2
log2_rowvector = tlog2
log2_matrix = tlog2
log2_array = tlog2

# R log10(T x)
# base-10 logarithm of x
from math import log10 as mlog10
from jax.numpy import log10 as tlog10

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

# R pow(T1 x, T2 y)
# Vectorized implementation of the pow function
pow_vectorized = lambda x, y: x ** y
pow_vector_vector = pow_vectorized
pow_rowvector_rowvector = pow_vectorized
pow_matrix_matrix = pow_vectorized
pow_array_array = pow_vectorized

pow_real_vector = pow_vectorized
pow_real_rowvector = pow_vectorized
pow_real_matrix = pow_vectorized
pow_real_array = pow_vectorized
pow_vector_real = pow_vectorized
pow_rowvector_real = pow_vectorized
pow_matrix_real = pow_vectorized
pow_array_real = pow_vectorized

pow_int_vector = _to_float_1(pow_vectorized)
pow_int_rowvector = _to_float_1(pow_vectorized)
pow_int_matrix = _to_float_1(pow_vectorized)
pow_int_array = _to_float_1(pow_vectorized)
pow_vector_int = _to_float_2(pow_vectorized)
pow_rowvector_int = _to_float_2(pow_vectorized)
pow_matrix_int = _to_float_2(pow_vectorized)
pow_array_int = _to_float_2(pow_vectorized)

# R inv(T x)
# inverse of x
inv_int = _XXX_TODO_XXX_("inv")
inv_real = _XXX_TODO_XXX_("inv")
inv_vector = _XXX_TODO_XXX_("inv")
inv_rowvector = _XXX_TODO_XXX_("inv")
inv_matrix = _XXX_TODO_XXX_("inv")
inv_array = _XXX_TODO_XXX_("inv")

# R inv_sqrt(T x)
# inverse of the square root of x
inv_sqrt_int = _XXX_TODO_XXX_("inv_sqrt")
inv_sqrt_real = _XXX_TODO_XXX_("inv_sqrt")
inv_sqrt_vector = _XXX_TODO_XXX_("inv_sqrt")
inv_sqrt_rowvector = _XXX_TODO_XXX_("inv_sqrt")
inv_sqrt_matrix = _XXX_TODO_XXX_("inv_sqrt")
inv_sqrt_array = _XXX_TODO_XXX_("inv_sqrt")

# R inv_square(T x)
# inverse of the square of x
inv_square_int = _XXX_TODO_XXX_("inv_square")
inv_square_real = _XXX_TODO_XXX_("inv_square")
inv_square_vector = _XXX_TODO_XXX_("inv_square")
inv_square_rowvector = _XXX_TODO_XXX_("inv_square")
inv_square_matrix = _XXX_TODO_XXX_("inv_square")
inv_square_array = _XXX_TODO_XXX_("inv_square")

## 3.9 Trigonometric Functions

# real hypot(real x, real y)
# Return the length of the hypotenuse of a right triangle with sides of length x and y.
from math import hypot

hypot_real_real = hypot

# R cos(T x)
# cosine of the angle x (in radians)
from math import cos as mcos
from jax.numpy import cos as tcos

cos_int = lambda x: array(mcos(x), dtype=dtype_float)
cos_real = tcos
cos_vector = tcos
cos_rowvector = tcos
cos_matrix = tcos
cos_array = tcos

# R sin(T x)
# sine of the angle x (in radians)
from math import sin as msin
from jax.numpy import sin as tsin

sin_int = lambda x: array(msin(x), dtype=dtype_float)
sin_real = tsin
sin_vector = tsin
sin_rowvector = tsin
sin_matrix = tsin
sin_array = tsin

# R tan(T x)
# tangent of the angle x (in radians)
from math import tan as mtan
from jax.numpy import tan as ttan

tan_int = lambda x: array(mtan(x), dtype=dtype_float)
tan_real = ttan
tan_vector = ttan
tan_rowvector = ttan
tan_matrix = ttan
tan_array = ttan

# R acos(T x)
# principal arc (inverse) cosine (in radians) of x
from math import acos as macos
from jax.numpy import arccos as tacos

acos_int = lambda x: array(macos(x), dtype=dtype_float)
acos_real = tacos
acos_vector = tacos
acos_rowvector = tacos
acos_matrix = tacos
acos_array = tacos

# R asin(T x)
# principal arc (inverse) sine (in radians) of x
from math import asin as masin
from jax.numpy import arcsin as tasin

asin_int = lambda x: array(masin(x), dtype=dtype_float)
asin_real = tasin
asin_vector = tasin
asin_rowvector = tasin
asin_matrix = tasin
asin_array = tasin

# R atan(T x)
# principal arc (inverse) tangent (in radians) of x, with values from −π
# to π
from math import atan as matan
from jax.numpy import arctan as tatan

atan_int = lambda x: array(matan(x), dtype=dtype_float)
atan_real = tatan
atan_vector = tatan
atan_rowvector = tatan
atan_matrix = tatan
atan_array = tatan

# real atan2(real y, real x)
# Return the principal arc (inverse) tangent (in radians) of y divided by x
from jax.numpy import arctan2 as tatan2

atan2_real_real = tatan2
atan2_int_real = _to_float_1(tatan2)
atan2_real_int = _to_float_2(tatan2)
atan2_int_int = _to_float_1_2(tatan2)

## 3.10 Hyperbolic Trigonometric Functions

# R cosh(T x)
# hyperbolic cosine of x (in radians)
from math import cosh as mcosh
from jax.numpy import cosh as tcosh

cosh_int = lambda x: array(mcosh(x), dtype=dtype_float)
cosh_real = tcosh
cosh_vector = tcosh
cosh_rowvector = tcosh
cosh_matrix = tcosh
cosh_array = tcosh

# R sinh(T x)
# hyperbolic sine of x (in radians)
from math import sinh as msinh
from jax.numpy import sinh as tsinh

sinh_int = lambda x: array(msinh(x), dtype=dtype_float)
sinh_real = tsinh
sinh_vector = tsinh
sinh_rowvector = tsinh
sinh_matrix = tsinh
sinh_array = tsinh

# R tanh(T x)
# hyperbolic tangent of x (in radians)
from math import tanh as mtanh
from jax.numpy import tanh as ttanh

tanh_int = lambda x: array(mtanh(x), dtype=dtype_float)
tanh_real = ttanh
tanh_vector = ttanh
tanh_rowvector = ttanh
tanh_matrix = ttanh
tanh_array = ttanh

# R acosh(T x)
# inverse hyperbolic cosine (in radians)
from math import acosh as macosh
from jax.numpy import arccosh as tacosh

acosh_int = lambda x: array(macosh(x), dtype=dtype_float)
acosh_real = tacosh
acosh_vector = tacosh
acosh_rowvector = tacosh
acosh_matrix = tacosh
acosh_array = tacosh

# R asinh(T x)
# inverse hyperbolic cosine (in radians)
from math import asinh as masinh
from jax.numpy import arcsinh as tasinh

asinh_int = lambda x: array(masinh(x), dtype=dtype_float)
asinh_real = tasinh
asinh_vector = tasinh
asinh_rowvector = tasinh
asinh_matrix = tasinh
asinh_array = tasinh

# R atanh(T x)
# inverse hyperbolic tangent (in radians) of x
from math import atanh as matanh
from jax.numpy import arctanh as tatanh

atanh_int = lambda x: array(matanh(x), dtype=dtype_float)
atanh_real = tatanh
atanh_vector = tatanh
atanh_rowvector = tatanh
atanh_matrix = tatanh
atanh_array = tatanh


## 3.11 Link Functions

# R logit(T x)
# log odds, or logit, function applied to x
logit_int = _XXX_TODO_XXX_("logit")
logit_real = _XXX_TODO_XXX_("logit")
logit_vector = _XXX_TODO_XXX_("logit")
logit_rowvector = _XXX_TODO_XXX_("logit")
logit_matrix = _XXX_TODO_XXX_("logit")
logit_array = _XXX_TODO_XXX_("logit")

# R inv_logit(T x)
# logistic sigmoid function applied to x
from jax.nn import sigmoid

inv_logit_int = lambda x: sigmoid(array(x, dtype=dtype_float))
inv_logit_real = sigmoid
inv_logit_vector = sigmoid
inv_logit_rowvector = sigmoid
inv_logit_matrix = sigmoid
inv_logit_array = sigmoid

# R inv_cloglog(T x)
# inverse of the complementary log-log function applied to x
inv_cloglog_int = _XXX_TODO_XXX_("inv_cloglog")
inv_cloglog_real = _XXX_TODO_XXX_("inv_cloglog")
inv_cloglog_vector = _XXX_TODO_XXX_("inv_cloglog")
inv_cloglog_rowvector = _XXX_TODO_XXX_("inv_cloglog")
inv_cloglog_matrix = _XXX_TODO_XXX_("inv_cloglog")
inv_cloglog_array = _XXX_TODO_XXX_("inv_cloglog")

## 3.12 Probability-related functions

## 3.12.1 Normal cumulative distribution functions

# R erf(T x)
# error function, also known as the Gauss error function, of x
erf_int = _XXX_TODO_XXX_("erf")
erf_real = _XXX_TODO_XXX_("erf")
erf_vector = _XXX_TODO_XXX_("erf")
erf_rowvector = _XXX_TODO_XXX_("erf")
erf_matrix = _XXX_TODO_XXX_("erf")
erf_array = _XXX_TODO_XXX_("erf")

# R erfc(T x)
# complementary error function of x
erfc_int = _XXX_TODO_XXX_("erfc")
erfc_real = _XXX_TODO_XXX_("erfc")
erfc_vector = _XXX_TODO_XXX_("erfc")
erfc_rowvector = _XXX_TODO_XXX_("erfc")
erfc_matrix = _XXX_TODO_XXX_("erfc")
erfc_array = _XXX_TODO_XXX_("erfc")

# R Phi(T x)
# standard normal cumulative distribution function of x
Phi_int = _XXX_TODO_XXX_("Phi")
Phi_real = _XXX_TODO_XXX_("Phi")
Phi_vector = _XXX_TODO_XXX_("Phi")
Phi_rowvector = _XXX_TODO_XXX_("Phi")
Phi_matrix = _XXX_TODO_XXX_("Phi")
Phi_array = _XXX_TODO_XXX_("Phi")

# R inv_Phi(T x)
# standard normal inverse cumulative distribution function of p, otherwise known as the quantile function
inv_Phi_int = _XXX_TODO_XXX_("inv_Phi")
inv_Phi_real = _XXX_TODO_XXX_("inv_Phi")
inv_Phi_vector = _XXX_TODO_XXX_("inv_Phi")
inv_Phi_rowvector = _XXX_TODO_XXX_("inv_Phi")
inv_Phi_matrix = _XXX_TODO_XXX_("inv_Phi")
inv_Phi_array = _XXX_TODO_XXX_("inv_Phi")

# R Phi_approx(T x)
# fast approximation of the unit (may replace Phi for probit regression with maximum absolute error of 0.00014, see (Bowling et al. 2009) for details)
Phi_approx_int = _XXX_TODO_XXX_("Phi_approx")
Phi_approx_real = _XXX_TODO_XXX_("Phi_approx")
Phi_approx_vector = _XXX_TODO_XXX_("Phi_approx")
Phi_approx_rowvector = _XXX_TODO_XXX_("Phi_approx")
Phi_approx_matrix = _XXX_TODO_XXX_("Phi_approx")
Phi_approx_array = _XXX_TODO_XXX_("Phi_approx")

## 3.12.2 Other probability-related functions

# real binary_log_loss(int y, real y_hat)
# Return the log loss function for for predicting ^y∈[0,1] for boolean outcome y∈{0,1}.
binary_log_loss_real_real = _XXX_TODO_XXX_("binary_log_loss")
binary_log_loss_int_real = _XXX_TODO_XXX_("binary_log_loss")
binary_log_loss_real_int = _XXX_TODO_XXX_("binary_log_loss")
binary_log_loss_int_int = _XXX_TODO_XXX_("binary_log_loss")

# R binary_log_loss(T1 x, T2 y)
# Vectorized implementation of the binary_log_loss function
binary_log_loss_vectorized = _XXX_TODO_XXX_("binary_log_loss")
binary_log_loss_vector_vector = binary_log_loss_vectorized
binary_log_loss_rowvector_rowvector = binary_log_loss_vectorized
binary_log_loss_matrix_matrix = binary_log_loss_vectorized
binary_log_loss_array_array = binary_log_loss_vectorized

binary_log_loss_real_vector = binary_log_loss_vectorized
binary_log_loss_real_rowvector = binary_log_loss_vectorized
binary_log_loss_real_matrix = binary_log_loss_vectorized
binary_log_loss_real_array = binary_log_loss_vectorized
binary_log_loss_vector_real = binary_log_loss_vectorized
binary_log_loss_rowvector_real = binary_log_loss_vectorized
binary_log_loss_matrix_real = binary_log_loss_vectorized
binary_log_loss_array_real = binary_log_loss_vectorized

binary_log_loss_int_vector = _to_float_1(binary_log_loss_vectorized)
binary_log_loss_int_rowvector = _to_float_1(binary_log_loss_vectorized)
binary_log_loss_int_matrix = _to_float_1(binary_log_loss_vectorized)
binary_log_loss_int_array = _to_float_1(binary_log_loss_vectorized)
binary_log_loss_vector_int = _to_float_2(binary_log_loss_vectorized)
binary_log_loss_rowvector_int = _to_float_2(binary_log_loss_vectorized)
binary_log_loss_matrix_int = _to_float_2(binary_log_loss_vectorized)
binary_log_loss_array_int = _to_float_2(binary_log_loss_vectorized)

# real owens_t(real h, real a)
# Return the Owen’s T function for the probability of the event X>h and 0<Y<aX where X and Y are independent standard normal random variables.
owens_t_real_real = _XXX_TODO_XXX_("owens_t")
owens_t_int_real = _XXX_TODO_XXX_("owens_t")
owens_t_real_int = _XXX_TODO_XXX_("owens_t")
owens_t_int_int = _XXX_TODO_XXX_("owens_t")

# R owens_t(T1 x, T2 y)
# Vectorized implementation of the owens_t function
owens_t_vectorized = _XXX_TODO_XXX_("owens_t")
owens_t_vector_vector = owens_t_vectorized
owens_t_rowvector_rowvector = owens_t_vectorized
owens_t_matrix_matrix = owens_t_vectorized
owens_t_array_array = owens_t_vectorized

owens_t_real_vector = owens_t_vectorized
owens_t_real_rowvector = owens_t_vectorized
owens_t_real_matrix = owens_t_vectorized
owens_t_real_array = owens_t_vectorized
owens_t_vector_real = owens_t_vectorized
owens_t_rowvector_real = owens_t_vectorized
owens_t_matrix_real = owens_t_vectorized
owens_t_array_real = owens_t_vectorized

owens_t_int_vector = _to_float_1(owens_t_vectorized)
owens_t_int_rowvector = _to_float_1(owens_t_vectorized)
owens_t_int_matrix = _to_float_1(owens_t_vectorized)
owens_t_int_array = _to_float_1(owens_t_vectorized)
owens_t_vector_int = _to_float_2(owens_t_vectorized)
owens_t_rowvector_int = _to_float_2(owens_t_vectorized)
owens_t_matrix_int = _to_float_2(owens_t_vectorized)
owens_t_array_int = _to_float_2(owens_t_vectorized)

## 3.13 Combinatorial functions

# real beta(real alpha, real beta)
# Return the beta function applied to alpha and beta. The beta function, B(α,β), computes the normalizing constant for the beta distribution, and is defined for α>0 and β>0. See section appendix for definition of B(α,β).
beta_real_real = _XXX_TODO_XXX_("beta")
beta_int_real = _XXX_TODO_XXX_("beta")
beta_real_int = _XXX_TODO_XXX_("beta")
beta_int_int = _XXX_TODO_XXX_("beta")

# R beta(T1 x, T2 y)
# Vectorized implementation of the beta function
beta_vectorized = _XXX_TODO_XXX_("beta")
beta_vector_vector = beta_vectorized
beta_rowvector_rowvector = beta_vectorized
beta_matrix_matrix = beta_vectorized
beta_array_array = beta_vectorized

beta_real_vector = beta_vectorized
beta_real_rowvector = beta_vectorized
beta_real_matrix = beta_vectorized
beta_real_array = beta_vectorized
beta_vector_real = beta_vectorized
beta_rowvector_real = beta_vectorized
beta_matrix_real = beta_vectorized
beta_array_real = beta_vectorized

beta_int_vector = _to_float_1(beta_vectorized)
beta_int_rowvector = _to_float_1(beta_vectorized)
beta_int_matrix = _to_float_1(beta_vectorized)
beta_int_array = _to_float_1(beta_vectorized)
beta_vector_int = _to_float_2(beta_vectorized)
beta_rowvector_int = _to_float_2(beta_vectorized)
beta_matrix_int = _to_float_2(beta_vectorized)
beta_array_int = _to_float_2(beta_vectorized)

# real inc_beta(real alpha, real beta, real x)
# Return the regularized incomplete beta function up to x applied to alpha and beta. See section appendix for a definition.
inc_beta_real_real_real = _XXX_TODO_XXX_("inc_beta")
# XXX TODO: lifting to other types XXX

# real lbeta(real alpha, real beta)
# Return the natural logarithm of the beta function applied to alpha and beta. The beta function, B(α,β), computes the normalizing constant for the beta distribution, and is defined for α>0 and β>0. lbeta(α,β)=logΓ(a)+logΓ(b)−logΓ(a+b) See section appendix for definition of B(α,β).
lbeta_real_real = _XXX_TODO_XXX_("lbeta")
lbeta_int_real = _XXX_TODO_XXX_("lbeta")
lbeta_real_int = _XXX_TODO_XXX_("lbeta")
lbeta_int_int = _XXX_TODO_XXX_("lbeta")

# R lbeta(T1 x, T2 y)
# Vectorized implementation of the lbeta function
lbeta_vectorized = _XXX_TODO_XXX_("lbeta")
lbeta_vector_vector = lbeta_vectorized
lbeta_rowvector_rowvector = lbeta_vectorized
lbeta_matrix_matrix = lbeta_vectorized
lbeta_array_array = lbeta_vectorized

lbeta_real_vector = lbeta_vectorized
lbeta_real_rowvector = lbeta_vectorized
lbeta_real_matrix = lbeta_vectorized
lbeta_real_array = lbeta_vectorized
lbeta_vector_real = lbeta_vectorized
lbeta_rowvector_real = lbeta_vectorized
lbeta_matrix_real = lbeta_vectorized
lbeta_array_real = lbeta_vectorized

lbeta_int_vector = _to_float_1(lbeta_vectorized)
lbeta_int_rowvector = _to_float_1(lbeta_vectorized)
lbeta_int_matrix = _to_float_1(lbeta_vectorized)
lbeta_int_array = _to_float_1(lbeta_vectorized)
lbeta_vector_int = _to_float_2(lbeta_vectorized)
lbeta_rowvector_int = _to_float_2(lbeta_vectorized)
lbeta_matrix_int = _to_float_2(lbeta_vectorized)
lbeta_array_int = _to_float_2(lbeta_vectorized)

# R tgamma(T x)
# gamma function applied to x. The gamma function is the generalization of the factorial function to continuous variables, defined so that Γ(n+1)=n!. See for a full definition of Γ(x). The function is defined for positive numbers and non-integral negative numbers,
tgamma_int = _XXX_TODO_XXX_("tgamma")
tgamma_real = _XXX_TODO_XXX_("tgamma")
tgamma_vector = _XXX_TODO_XXX_("tgamma")
tgamma_rowvector = _XXX_TODO_XXX_("tgamma")
tgamma_matrix = _XXX_TODO_XXX_("tgamma")
tgamma_array = _XXX_TODO_XXX_("tgamma")

# R lgamma(T x)
# natural logarithm of the gamma function applied to x,
lgamma_int = _XXX_TODO_XXX_("lgamma")
lgamma_real = _XXX_TODO_XXX_("lgamma")
lgamma_vector = _XXX_TODO_XXX_("lgamma")
lgamma_rowvector = _XXX_TODO_XXX_("lgamma")
lgamma_matrix = _XXX_TODO_XXX_("lgamma")
lgamma_array = _XXX_TODO_XXX_("lgamma")

# R digamma(T x)
# digamma function applied to x. The digamma function is the derivative of the natural logarithm of the Gamma function. The function is defined for positive numbers and non-integral negative numbers
digamma_int = _XXX_TODO_XXX_("digamma")
digamma_real = _XXX_TODO_XXX_("digamma")
digamma_vector = _XXX_TODO_XXX_("digamma")
digamma_rowvector = _XXX_TODO_XXX_("digamma")
digamma_matrix = _XXX_TODO_XXX_("digamma")
digamma_array = _XXX_TODO_XXX_("digamma")

# R trigamma(T x)
# trigamma function applied to x. The trigamma function is the second derivative of the natural logarithm of the Gamma function
trigamma_int = _XXX_TODO_XXX_("trigamma")
trigamma_real = _XXX_TODO_XXX_("trigamma")
trigamma_vector = _XXX_TODO_XXX_("trigamma")
trigamma_rowvector = _XXX_TODO_XXX_("trigamma")
trigamma_matrix = _XXX_TODO_XXX_("trigamma")
trigamma_array = _XXX_TODO_XXX_("trigamma")

# real lmgamma(int n, real x)
# Return the natural logarithm of the multivariate gamma function Γn
# with n dimensions applied to x.
lmgamma_real_real = _XXX_TODO_XXX_("lmgamma")
lmgamma_int_real = _XXX_TODO_XXX_("lmgamma")
lmgamma_real_int = _XXX_TODO_XXX_("lmgamma")
lmgamma_int_int = _XXX_TODO_XXX_("lmgamma")

# R lmgamma(T1 x, T2 y)
# Vectorized implementation of the lmgamma function
lmgamma_vectorized = _XXX_TODO_XXX_("lmgamma")
lmgamma_vector_vector = lmgamma_vectorized
lmgamma_rowvector_rowvector = lmgamma_vectorized
lmgamma_matrix_matrix = lmgamma_vectorized
lmgamma_array_array = lmgamma_vectorized

lmgamma_real_vector = lmgamma_vectorized
lmgamma_real_rowvector = lmgamma_vectorized
lmgamma_real_matrix = lmgamma_vectorized
lmgamma_real_array = lmgamma_vectorized
lmgamma_vector_real = lmgamma_vectorized
lmgamma_rowvector_real = lmgamma_vectorized
lmgamma_matrix_real = lmgamma_vectorized
lmgamma_array_real = lmgamma_vectorized

lmgamma_int_vector = _to_float_1(lmgamma_vectorized)
lmgamma_int_rowvector = _to_float_1(lmgamma_vectorized)
lmgamma_int_matrix = _to_float_1(lmgamma_vectorized)
lmgamma_int_array = _to_float_1(lmgamma_vectorized)
lmgamma_vector_int = _to_float_2(lmgamma_vectorized)
lmgamma_rowvector_int = _to_float_2(lmgamma_vectorized)
lmgamma_matrix_int = _to_float_2(lmgamma_vectorized)
lmgamma_array_int = _to_float_2(lmgamma_vectorized)

# real gamma_p(real a, real z)
# Return the normalized lower incomplete gamma function of a and z defined for positive a and nonnegative z.
lmgamma_real_real = _XXX_TODO_XXX_("lmgamma")
lmgamma_int_real = _to_float_1(lmgamma_real_real)
lmgamma_real_int = _to_float_2(lmgamma_real_real)
lmgamma_int_int = _to_float_1_2(lmgamma_real_real)

# R gamma_p(T1 x, T2 y)
# Vectorized implementation of the gamma_p function
lmgamma_vectorized = _XXX_TODO_XXX_("lmgamma")
lmgamma_vector_vector = lmgamma_vectorized
lmgamma_rowvector_rowvector = lmgamma_vectorized
lmgamma_matrix_matrix = lmgamma_vectorized
lmgamma_array_array = lmgamma_vectorized

lmgamma_real_vector = lmgamma_vectorized
lmgamma_real_rowvector = lmgamma_vectorized
lmgamma_real_matrix = lmgamma_vectorized
lmgamma_real_array = lmgamma_vectorized
lmgamma_vector_real = lmgamma_vectorized
lmgamma_rowvector_real = lmgamma_vectorized
lmgamma_matrix_real = lmgamma_vectorized
lmgamma_array_real = lmgamma_vectorized

lmgamma_int_vector = _to_float_1(lmgamma_vectorized)
lmgamma_int_rowvector = _to_float_1(lmgamma_vectorized)
lmgamma_int_matrix = _to_float_1(lmgamma_vectorized)
lmgamma_int_array = _to_float_1(lmgamma_vectorized)
lmgamma_vector_int = _to_float_2(lmgamma_vectorized)
lmgamma_rowvector_int = _to_float_2(lmgamma_vectorized)
lmgamma_matrix_int = _to_float_2(lmgamma_vectorized)
lmgamma_array_int = _to_float_2(lmgamma_vectorized)

# real gamma_q(real a, real z)
# Return the normalized upper incomplete gamma function of a and z defined for positive a and nonnegative z.
gamma_q_real_real = _XXX_TODO_XXX_("gamma_q")
gamma_q_int_real = _to_float_1(gamma_q_real_real)
gamma_q_real_int = _to_float_2(gamma_q_real_real)
gamma_q_int_int = _to_float_1_2(gamma_q_real_real)

# R gamma_q(T1 x, T2 y)
# Vectorized implementation of the gamma_q function
gamma_q_vectorized = _XXX_TODO_XXX_("gamma_q")
gamma_q_vector_vector = gamma_q_vectorized
gamma_q_rowvector_rowvector = gamma_q_vectorized
gamma_q_matrix_matrix = gamma_q_vectorized
gamma_q_array_array = gamma_q_vectorized

gamma_q_real_vector = gamma_q_vectorized
gamma_q_real_rowvector = gamma_q_vectorized
gamma_q_real_matrix = gamma_q_vectorized
gamma_q_real_array = gamma_q_vectorized
gamma_q_vector_real = gamma_q_vectorized
gamma_q_rowvector_real = gamma_q_vectorized
gamma_q_matrix_real = gamma_q_vectorized
gamma_q_array_real = gamma_q_vectorized

gamma_q_int_vector = _to_float_1(gamma_q_vectorized)
gamma_q_int_rowvector = _to_float_1(gamma_q_vectorized)
gamma_q_int_matrix = _to_float_1(gamma_q_vectorized)
gamma_q_int_array = _to_float_1(gamma_q_vectorized)
gamma_q_vector_int = _to_float_2(gamma_q_vectorized)
gamma_q_rowvector_int = _to_float_2(gamma_q_vectorized)
gamma_q_matrix_int = _to_float_2(gamma_q_vectorized)
gamma_q_array_int = _to_float_2(gamma_q_vectorized)

# real binomial_coefficient_log(real x, real y)
# Warning: This function is deprecated and should be replaced with lchoose. Return the natural logarithm of the binomial coefficient of x and y. For non-negative integer inputs, the binomial coefficient function is written as (xy)
# and pronounced “x choose y.” This function generalizes to real numbers using the gamma function. For 0≤y≤x, binomial_coefficient_log(x,y)=logΓ(x+1)−logΓ(y+1)−logΓ(x−y+1).
binomial_coefficient_log_real_real = _XXX_TODO_XXX_("binomial_coefficient_log")
binomial_coefficient_log_int_real = _to_float_1(binomial_coefficient_log_real_real)
binomial_coefficient_log_real_int = _to_float_2(binomial_coefficient_log_real_real)
binomial_coefficient_log_int_int = _to_float_1_2(binomial_coefficient_log_real_real)

# R binomial_coefficient_log(T1 x, T2 y)
# Vectorized implementation of the binomial_coefficient_log function
binomial_coefficient_log_vectorized = _XXX_TODO_XXX_("binomial_coefficient_log")
binomial_coefficient_log_vector_vector = binomial_coefficient_log_vectorized
binomial_coefficient_log_rowvector_rowvector = binomial_coefficient_log_vectorized
binomial_coefficient_log_matrix_matrix = binomial_coefficient_log_vectorized
binomial_coefficient_log_array_array = binomial_coefficient_log_vectorized

binomial_coefficient_log_real_vector = binomial_coefficient_log_vectorized
binomial_coefficient_log_real_rowvector = binomial_coefficient_log_vectorized
binomial_coefficient_log_real_matrix = binomial_coefficient_log_vectorized
binomial_coefficient_log_real_array = binomial_coefficient_log_vectorized
binomial_coefficient_log_vector_real = binomial_coefficient_log_vectorized
binomial_coefficient_log_rowvector_real = binomial_coefficient_log_vectorized
binomial_coefficient_log_matrix_real = binomial_coefficient_log_vectorized
binomial_coefficient_log_array_real = binomial_coefficient_log_vectorized

binomial_coefficient_log_int_vector = _to_float_1(binomial_coefficient_log_vectorized)
binomial_coefficient_log_int_rowvector = _to_float_1(binomial_coefficient_log_vectorized)
binomial_coefficient_log_int_matrix = _to_float_1(binomial_coefficient_log_vectorized)
binomial_coefficient_log_int_array = _to_float_1(binomial_coefficient_log_vectorized)
binomial_coefficient_log_vector_int = _to_float_2(binomial_coefficient_log_vectorized)
binomial_coefficient_log_rowvector_int = _to_float_2(binomial_coefficient_log_vectorized)
binomial_coefficient_log_matrix_int = _to_float_2(binomial_coefficient_log_vectorized)
binomial_coefficient_log_array_int = _to_float_2(binomial_coefficient_log_vectorized)

# int choose(int x, int y)
# Return the binomial coefficient of x and y. For non-negative integer inputs, the binomial coefficient function is written as (xy)
# and pronounced “x choose y.” In its the antilog of the lchoose function but returns an integer rather than a real number with no non-zero decimal places. For 0≤y≤x, the binomial coefficient function can be defined via the factorial function choose(x,y)=x!(y!)(x−y)!.
choose_real_real = _XXX_TODO_XXX_("choose")
choose_int_real = _to_float_1(choose_real_real)
choose_real_int = _to_float_2(choose_real_real)
choose_int_int = _to_float_1_2(choose_real_real)

# R choose(T1 x, T2 y)
# Vectorized implementation of the choose function
choose_vectorized = _XXX_TODO_XXX_("choose")
choose_vector_vector = choose_vectorized
choose_rowvector_rowvector = choose_vectorized
choose_matrix_matrix = choose_vectorized
choose_array_array = choose_vectorized

choose_real_vector = choose_vectorized
choose_real_rowvector = choose_vectorized
choose_real_matrix = choose_vectorized
choose_real_array = choose_vectorized
choose_vector_real = choose_vectorized
choose_rowvector_real = choose_vectorized
choose_matrix_real = choose_vectorized
choose_array_real = choose_vectorized

choose_int_vector = _to_float_1(choose_vectorized)
choose_int_rowvector = _to_float_1(choose_vectorized)
choose_int_matrix = _to_float_1(choose_vectorized)
choose_int_array = _to_float_1(choose_vectorized)
choose_vector_int = _to_float_2(choose_vectorized)
choose_rowvector_int = _to_float_2(choose_vectorized)
choose_matrix_int = _to_float_2(choose_vectorized)
choose_array_int = _to_float_2(choose_vectorized)

# real bessel_first_kind(int v, real x)
# Return the Bessel function of the first kind with order v applied to x. bessel_first_kind(v,x)=Jv(x), where Jv(x)=(12x)v∞∑k=0(−14x2)kk!Γ(v+k+1)
bessel_first_kind_real_real = _XXX_TODO_XXX_("bessel_first_kind")
bessel_first_kind_int_real = _to_float_1(bessel_first_kind_real_real)
bessel_first_kind_real_int = _to_float_2(bessel_first_kind_real_real)
bessel_first_kind_int_int = _to_float_1_2(bessel_first_kind_real_real)

# R bessel_first_kind(T1 x, T2 y)
# Vectorized implementation of the bessel_first_kind function
bessel_first_kind_vectorized = _XXX_TODO_XXX_("bessel_first_kind")
bessel_first_kind_vector_vector = bessel_first_kind_vectorized
bessel_first_kind_rowvector_rowvector = bessel_first_kind_vectorized
bessel_first_kind_matrix_matrix = bessel_first_kind_vectorized
bessel_first_kind_array_array = bessel_first_kind_vectorized

bessel_first_kind_real_vector = bessel_first_kind_vectorized
bessel_first_kind_real_rowvector = bessel_first_kind_vectorized
bessel_first_kind_real_matrix = bessel_first_kind_vectorized
bessel_first_kind_real_array = bessel_first_kind_vectorized
bessel_first_kind_vector_real = bessel_first_kind_vectorized
bessel_first_kind_rowvector_real = bessel_first_kind_vectorized
bessel_first_kind_matrix_real = bessel_first_kind_vectorized
bessel_first_kind_array_real = bessel_first_kind_vectorized

bessel_first_kind_int_vector = _to_float_1(bessel_first_kind_vectorized)
bessel_first_kind_int_rowvector = _to_float_1(bessel_first_kind_vectorized)
bessel_first_kind_int_matrix = _to_float_1(bessel_first_kind_vectorized)
bessel_first_kind_int_array = _to_float_1(bessel_first_kind_vectorized)
bessel_first_kind_vector_int = _to_float_2(bessel_first_kind_vectorized)
bessel_first_kind_rowvector_int = _to_float_2(bessel_first_kind_vectorized)
bessel_first_kind_matrix_int = _to_float_2(bessel_first_kind_vectorized)
bessel_first_kind_array_int = _to_float_2(bessel_first_kind_vectorized)

# real bessel_second_kind(int v, real x)
# Return the Bessel function of the second kind with order v applied to x defined for positive x and v. For x,v>0, bessel_second_kind(v,x)={Yv(x)if x>0errorotherwise where Yv(x)=Jv(x)cos(vπ)−J−v(x)sin(vπ)
bessel_second_kind_real_real = _XXX_TODO_XXX_("bessel_second_kind")
bessel_second_kind_int_real = _to_float_1(bessel_second_kind_real_real)
bessel_second_kind_real_int = _to_float_2(bessel_second_kind_real_real)
bessel_second_kind_int_int = _to_float_1_2(bessel_second_kind_real_real)

# R bessel_second_kind(T1 x, T2 y)
# Vectorized implementation of the bessel_second_kind function
bessel_second_kind_vectorized = _XXX_TODO_XXX_("bessel_second_kind")
bessel_second_kind_vector_vector = bessel_second_kind_vectorized
bessel_second_kind_rowvector_rowvector = bessel_second_kind_vectorized
bessel_second_kind_matrix_matrix = bessel_second_kind_vectorized
bessel_second_kind_array_array = bessel_second_kind_vectorized

bessel_second_kind_real_vector = bessel_second_kind_vectorized
bessel_second_kind_real_rowvector = bessel_second_kind_vectorized
bessel_second_kind_real_matrix = bessel_second_kind_vectorized
bessel_second_kind_real_array = bessel_second_kind_vectorized
bessel_second_kind_vector_real = bessel_second_kind_vectorized
bessel_second_kind_rowvector_real = bessel_second_kind_vectorized
bessel_second_kind_matrix_real = bessel_second_kind_vectorized
bessel_second_kind_array_real = bessel_second_kind_vectorized

bessel_second_kind_int_vector = _to_float_1(bessel_second_kind_vectorized)
bessel_second_kind_int_rowvector = _to_float_1(bessel_second_kind_vectorized)
bessel_second_kind_int_matrix = _to_float_1(bessel_second_kind_vectorized)
bessel_second_kind_int_array = _to_float_1(bessel_second_kind_vectorized)
bessel_second_kind_vector_int = _to_float_2(bessel_second_kind_vectorized)
bessel_second_kind_rowvector_int = _to_float_2(bessel_second_kind_vectorized)
bessel_second_kind_matrix_int = _to_float_2(bessel_second_kind_vectorized)
bessel_second_kind_array_int = _to_float_2(bessel_second_kind_vectorized)

# real modified_bessel_first_kind(int v, real z)
# Return the modified Bessel function of the first kind with order v applied to z defined for all z and integer v. modified_bessel_first_kind(v,z)=Iv(z)
# where Iv(z)=(12z)v∞∑k=0(14z2)kk!Γ(v+k+1)
modified_bessel_first_kind_real_real = _XXX_TODO_XXX_("modified_bessel_first_kind")
modified_bessel_first_kind_int_real = _to_float_1(modified_bessel_first_kind_real_real)
modified_bessel_first_kind_real_int = _to_float_2(modified_bessel_first_kind_real_real)
modified_bessel_first_kind_int_int = _to_float_1_2(modified_bessel_first_kind_real_real)

# R modified_bessel_first_kind(T1 x, T2 y)
# Vectorized implementation of the modified_bessel_first_kind function
modified_bessel_first_kind_vectorized = _XXX_TODO_XXX_("modified_bessel_first_kind")
modified_bessel_first_kind_vector_vector = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_rowvector_rowvector = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_matrix_matrix = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_array_array = modified_bessel_first_kind_vectorized

modified_bessel_first_kind_real_vector = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_real_rowvector = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_real_matrix = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_real_array = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_vector_real = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_rowvector_real = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_matrix_real = modified_bessel_first_kind_vectorized
modified_bessel_first_kind_array_real = modified_bessel_first_kind_vectorized

modified_bessel_first_kind_int_vector = _to_float_1(modified_bessel_first_kind_vectorized)
modified_bessel_first_kind_int_rowvector = _to_float_1(modified_bessel_first_kind_vectorized)
modified_bessel_first_kind_int_matrix = _to_float_1(modified_bessel_first_kind_vectorized)
modified_bessel_first_kind_int_array = _to_float_1(modified_bessel_first_kind_vectorized)
modified_bessel_first_kind_vector_int = _to_float_2(modified_bessel_first_kind_vectorized)
modified_bessel_first_kind_rowvector_int = _to_float_2(modified_bessel_first_kind_vectorized)
modified_bessel_first_kind_matrix_int = _to_float_2(modified_bessel_first_kind_vectorized)
modified_bessel_first_kind_array_int = _to_float_2(modified_bessel_first_kind_vectorized)

# real log_modified_bessel_first_kind(real v, real z)
# Return the log of the modified Bessel function of the first kind. v does not have to be an integer.
log_modified_bessel_first_kind_real_real = _XXX_TODO_XXX_("log_modified_bessel_first_kind")
log_modified_bessel_first_kind_int_real = _to_float_1(log_modified_bessel_first_kind_real_real)
log_modified_bessel_first_kind_real_int = _to_float_2(log_modified_bessel_first_kind_real_real)
log_modified_bessel_first_kind_int_int = _to_float_1_2(log_modified_bessel_first_kind_real_real)

# R log_modified_bessel_first_kind(T1 x, T2 y)
# Vectorized implementation of the log_modified_bessel_first_kind function
log_modified_bessel_first_kind_vectorized = _XXX_TODO_XXX_("log_modified_bessel_first_kind")
log_modified_bessel_first_kind_vector_vector = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_rowvector_rowvector = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_matrix_matrix = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_array_array = log_modified_bessel_first_kind_vectorized

log_modified_bessel_first_kind_real_vector = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_real_rowvector = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_real_matrix = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_real_array = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_vector_real = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_rowvector_real = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_matrix_real = log_modified_bessel_first_kind_vectorized
log_modified_bessel_first_kind_array_real = log_modified_bessel_first_kind_vectorized

log_modified_bessel_first_kind_int_vector = _to_float_1(log_modified_bessel_first_kind_vectorized)
log_modified_bessel_first_kind_int_rowvector = _to_float_1(log_modified_bessel_first_kind_vectorized)
log_modified_bessel_first_kind_int_matrix = _to_float_1(log_modified_bessel_first_kind_vectorized)
log_modified_bessel_first_kind_int_array = _to_float_1(log_modified_bessel_first_kind_vectorized)
log_modified_bessel_first_kind_vector_int = _to_float_2(log_modified_bessel_first_kind_vectorized)
log_modified_bessel_first_kind_rowvector_int = _to_float_2(log_modified_bessel_first_kind_vectorized)
log_modified_bessel_first_kind_matrix_int = _to_float_2(log_modified_bessel_first_kind_vectorized)
log_modified_bessel_first_kind_array_int = _to_float_2(log_modified_bessel_first_kind_vectorized)

# real modified_bessel_second_kind(int v, real z)
# Return the modified Bessel function of the second kind with order v applied to z defined for positive z and integer v. modified_bessel_second_kind(v,z)={Kv(z)if z>0errorif z≤0 where Kv(z)=π2⋅I−v(z)−Iv(z)sin(vπ)
modified_bessel_second_kind_real_real = _XXX_TODO_XXX_("modified_bessel_second_kind")
modified_bessel_second_kind_int_real = _to_float_1(modified_bessel_second_kind_real_real)
modified_bessel_second_kind_real_int = _to_float_2(modified_bessel_second_kind_real_real)
modified_bessel_second_kind_int_int = _to_float_1_2(modified_bessel_second_kind_real_real)

# R modified_bessel_second_kind(T1 x, T2 y)
# Vectorized implementation of the modified_bessel_second_kind function
modified_bessel_second_kind_vectorized = _XXX_TODO_XXX_("modified_bessel_second_kind")
modified_bessel_second_kind_vector_vector = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_rowvector_rowvector = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_matrix_matrix = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_array_array = modified_bessel_second_kind_vectorized

modified_bessel_second_kind_real_vector = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_real_rowvector = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_real_matrix = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_real_array = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_vector_real = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_rowvector_real = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_matrix_real = modified_bessel_second_kind_vectorized
modified_bessel_second_kind_array_real = modified_bessel_second_kind_vectorized

modified_bessel_second_kind_int_vector = _to_float_1(modified_bessel_second_kind_vectorized)
modified_bessel_second_kind_int_rowvector = _to_float_1(modified_bessel_second_kind_vectorized)
modified_bessel_second_kind_int_matrix = _to_float_1(modified_bessel_second_kind_vectorized)
modified_bessel_second_kind_int_array = _to_float_1(modified_bessel_second_kind_vectorized)
modified_bessel_second_kind_vector_int = _to_float_2(modified_bessel_second_kind_vectorized)
modified_bessel_second_kind_rowvector_int = _to_float_2(modified_bessel_second_kind_vectorized)
modified_bessel_second_kind_matrix_int = _to_float_2(modified_bessel_second_kind_vectorized)
modified_bessel_second_kind_array_int = _to_float_2(modified_bessel_second_kind_vectorized)

# real falling_factorial(real x, real n)
# Return the falling factorial of x with power n defined for positive x and real n. falling_factorial(x,n)={(x)nif x>0errorif x≤0 where (x)n=Γ(x+1)Γ(x−n+1)
falling_factorial_real_real = _XXX_TODO_XXX_("falling_factorial")
falling_factorial_int_real = _to_float_1(falling_factorial_real_real)
falling_factorial_real_int = _to_float_2(falling_factorial_real_real)
falling_factorial_int_int = _to_float_1_2(falling_factorial_real_real)

# R falling_factorial(T1 x, T2 y)
# Vectorized implementation of the falling_factorial function
falling_factorial_vectorized = _XXX_TODO_XXX_("falling_factorial")
falling_factorial_vector_vector = falling_factorial_vectorized
falling_factorial_rowvector_rowvector = falling_factorial_vectorized
falling_factorial_matrix_matrix = falling_factorial_vectorized
falling_factorial_array_array = falling_factorial_vectorized

falling_factorial_real_vector = falling_factorial_vectorized
falling_factorial_real_rowvector = falling_factorial_vectorized
falling_factorial_real_matrix = falling_factorial_vectorized
falling_factorial_real_array = falling_factorial_vectorized
falling_factorial_vector_real = falling_factorial_vectorized
falling_factorial_rowvector_real = falling_factorial_vectorized
falling_factorial_matrix_real = falling_factorial_vectorized
falling_factorial_array_real = falling_factorial_vectorized

falling_factorial_int_vector = _to_float_1(falling_factorial_vectorized)
falling_factorial_int_rowvector = _to_float_1(falling_factorial_vectorized)
falling_factorial_int_matrix = _to_float_1(falling_factorial_vectorized)
falling_factorial_int_array = _to_float_1(falling_factorial_vectorized)
falling_factorial_vector_int = _to_float_2(falling_factorial_vectorized)
falling_factorial_rowvector_int = _to_float_2(falling_factorial_vectorized)
falling_factorial_matrix_int = _to_float_2(falling_factorial_vectorized)
falling_factorial_array_int = _to_float_2(falling_factorial_vectorized)

# real lchoose(real x, real y)
# Return the natural logarithm of the generalized binomial coefficient of x and y. For non-negative integer inputs, the binomial coefficient function is written as (xy) and pronounced “x choose y.” This function generalizes to real numbers using the gamma function. For 0≤y≤x, binomial_coefficient_log(x,y)=logΓ(x+1)−logΓ(y+1)−logΓ(x−y+1).
lchoose_real_real = _XXX_TODO_XXX_("lchoose")
lchoose_int_real = _to_float_1(lchoose_real_real)
lchoose_real_int = _to_float_2(lchoose_real_real)
lchoose_int_int = _to_float_1_2(lchoose_real_real)

# real log_falling_factorial(real x, real n)
# Return the log of the falling factorial of x with power n defined for positive x and real n. log_falling_factorial(x,n)={log(x)nif x>0errorif x≤0
log_falling_factorial_real_real = _XXX_TODO_XXX_("log_falling_factorial")
log_falling_factorial_int_real = _to_float_1(log_falling_factorial_real_real)
log_falling_factorial_real_int = _to_float_2(log_falling_factorial_real_real)
log_falling_factorial_int_int = _to_float_1_2(log_falling_factorial_real_real)

# real rising_factorial(real x, int n)
# Return the rising factorial of x with power n defined for positive x and integer n. rising_factorial(x,n)={x(n)if x>0errorif x≤0
# where x(n)=Γ(x+n)Γ(x)
rising_factorial_real_real = _XXX_TODO_XXX_("rising_factorial")
rising_factorial_int_real = _to_float_1(rising_factorial_real_real)
rising_factorial_real_int = _to_float_2(rising_factorial_real_real)
rising_factorial_int_int = _to_float_1_2(rising_factorial_real_real)

# R rising_factorial(T1 x, T2 y)
# Vectorized implementation of the rising_factorial function
rising_factorial_vectorized = _XXX_TODO_XXX_("rising_factorial")
rising_factorial_vector_vector = rising_factorial_vectorized
rising_factorial_rowvector_rowvector = rising_factorial_vectorized
rising_factorial_matrix_matrix = rising_factorial_vectorized
rising_factorial_array_array = rising_factorial_vectorized

rising_factorial_real_vector = rising_factorial_vectorized
rising_factorial_real_rowvector = rising_factorial_vectorized
rising_factorial_real_matrix = rising_factorial_vectorized
rising_factorial_real_array = rising_factorial_vectorized
rising_factorial_vector_real = rising_factorial_vectorized
rising_factorial_rowvector_real = rising_factorial_vectorized
rising_factorial_matrix_real = rising_factorial_vectorized
rising_factorial_array_real = rising_factorial_vectorized

rising_factorial_int_vector = _to_float_1(rising_factorial_vectorized)
rising_factorial_int_rowvector = _to_float_1(rising_factorial_vectorized)
rising_factorial_int_matrix = _to_float_1(rising_factorial_vectorized)
rising_factorial_int_array = _to_float_1(rising_factorial_vectorized)
rising_factorial_vector_int = _to_float_2(rising_factorial_vectorized)
rising_factorial_rowvector_int = _to_float_2(rising_factorial_vectorized)
rising_factorial_matrix_int = _to_float_2(rising_factorial_vectorized)
rising_factorial_array_int = _to_float_2(rising_factorial_vectorized)

# real log_rising_factorial(real x, real n)
# Return the log of the rising factorial of x with power n defined for positive x and real n. log_rising_factorial(x,n)={logx(n)if x>0errorif x≤0
log_rising_factorial_real_real = _XXX_TODO_XXX_("log_rising_factorial")
log_rising_factorial_int_real = _to_float_1(log_rising_factorial_real_real)
log_rising_factorial_real_int = _to_float_2(log_rising_factorial_real_real)
log_rising_factorial_int_int = _to_float_1_2(log_rising_factorial_real_real)

# R log_rising_factorial(T1 x, T2 y)
# Vectorized implementation of the log_rising_factorial function
log_rising_factorial_vectorized = _XXX_TODO_XXX_("log_rising_factorial")
log_rising_factorial_vector_vector = log_rising_factorial_vectorized
log_rising_factorial_rowvector_rowvector = log_rising_factorial_vectorized
log_rising_factorial_matrix_matrix = log_rising_factorial_vectorized
log_rising_factorial_array_array = log_rising_factorial_vectorized

log_rising_factorial_real_vector = log_rising_factorial_vectorized
log_rising_factorial_real_rowvector = log_rising_factorial_vectorized
log_rising_factorial_real_matrix = log_rising_factorial_vectorized
log_rising_factorial_real_array = log_rising_factorial_vectorized
log_rising_factorial_vector_real = log_rising_factorial_vectorized
log_rising_factorial_rowvector_real = log_rising_factorial_vectorized
log_rising_factorial_matrix_real = log_rising_factorial_vectorized
log_rising_factorial_array_real = log_rising_factorial_vectorized

log_rising_factorial_int_vector = _to_float_1(log_rising_factorial_vectorized)
log_rising_factorial_int_rowvector = _to_float_1(log_rising_factorial_vectorized)
log_rising_factorial_int_matrix = _to_float_1(log_rising_factorial_vectorized)
log_rising_factorial_int_array = _to_float_1(log_rising_factorial_vectorized)
log_rising_factorial_vector_int = _to_float_2(log_rising_factorial_vectorized)
log_rising_factorial_rowvector_int = _to_float_2(log_rising_factorial_vectorized)
log_rising_factorial_matrix_int = _to_float_2(log_rising_factorial_vectorized)
log_rising_factorial_array_int = _to_float_2(log_rising_factorial_vectorized)

## 3.14 Composed Functions

# R expm1(T x)
# natural exponential of x minus 1
expm1_int = _XXX_TODO_XXX_("expm1")
expm1_real = _XXX_TODO_XXX_("expm1")
expm1_vector = _XXX_TODO_XXX_("expm1")
expm1_rowvector = _XXX_TODO_XXX_("expm1")
expm1_matrix = _XXX_TODO_XXX_("expm1")
expm1_array = _XXX_TODO_XXX_("expm1")

# real fma(real x, real y, real z)
# Return z plus the result of x multiplied by y. fma(x,y,z)=(x×y)+z
fma_real_real_real = _XXX_TODO_XXX_("fma")
# XXX TODO: lifting to other types XXX

# real multiply_log(real x, real y)
# Warning: This function is deprecated and should be replaced with lmultiply. Return the product of x and the natural logarithm of y.
multiply_log_real_real = _XXX_TODO_XXX_("multiply_log")
multiply_log_int_real = _to_float_1(multiply_log_real_real)
multiply_log_real_int = _to_float_2(multiply_log_real_real)
multiply_log_int_int = _to_float_1_2(multiply_log_real_real)

# R multiply_log(T1 x, T2 y)
# Vectorized implementation of the multiply_log function
multiply_log_vectorized = _XXX_TODO_XXX_("multiply_log")
multiply_log_vector_vector = multiply_log_vectorized
multiply_log_rowvector_rowvector = multiply_log_vectorized
multiply_log_matrix_matrix = multiply_log_vectorized
multiply_log_array_array = multiply_log_vectorized

multiply_log_real_vector = multiply_log_vectorized
multiply_log_real_rowvector = multiply_log_vectorized
multiply_log_real_matrix = multiply_log_vectorized
multiply_log_real_array = multiply_log_vectorized
multiply_log_vector_real = multiply_log_vectorized
multiply_log_rowvector_real = multiply_log_vectorized
multiply_log_matrix_real = multiply_log_vectorized
multiply_log_array_real = multiply_log_vectorized

multiply_log_int_vector = _to_float_1(multiply_log_vectorized)
multiply_log_int_rowvector = _to_float_1(multiply_log_vectorized)
multiply_log_int_matrix = _to_float_1(multiply_log_vectorized)
multiply_log_int_array = _to_float_1(multiply_log_vectorized)
multiply_log_vector_int = _to_float_2(multiply_log_vectorized)
multiply_log_rowvector_int = _to_float_2(multiply_log_vectorized)
multiply_log_matrix_int = _to_float_2(multiply_log_vectorized)
multiply_log_array_int = _to_float_2(multiply_log_vectorized)

# real ldexp(real x, int y)
# Return the product of x and two raised to the y power.
ldexp_real_real = _XXX_TODO_XXX_("ldexp")
ldexp_int_real = _to_float_1(ldexp_real_real)
ldexp_real_int = _to_float_2(ldexp_real_real)
ldexp_int_int = _to_float_1_2(ldexp_real_real)

# R ldexp(T1 x, T2 y)
# Vectorized implementation of the ldexp function
ldexp_vectorized = _XXX_TODO_XXX_("ldexp")
ldexp_vector_vector = ldexp_vectorized
ldexp_rowvector_rowvector = ldexp_vectorized
ldexp_matrix_matrix = ldexp_vectorized
ldexp_array_array = ldexp_vectorized

ldexp_real_vector = ldexp_vectorized
ldexp_real_rowvector = ldexp_vectorized
ldexp_real_matrix = ldexp_vectorized
ldexp_real_array = ldexp_vectorized
ldexp_vector_real = ldexp_vectorized
ldexp_rowvector_real = ldexp_vectorized
ldexp_matrix_real = ldexp_vectorized
ldexp_array_real = ldexp_vectorized

ldexp_int_vector = _to_float_1(ldexp_vectorized)
ldexp_int_rowvector = _to_float_1(ldexp_vectorized)
ldexp_int_matrix = _to_float_1(ldexp_vectorized)
ldexp_int_array = _to_float_1(ldexp_vectorized)
ldexp_vector_int = _to_float_2(ldexp_vectorized)
ldexp_rowvector_int = _to_float_2(ldexp_vectorized)
ldexp_matrix_int = _to_float_2(ldexp_vectorized)
ldexp_array_int = _to_float_2(ldexp_vectorized)

# real lmultiply(real x, real y)
# Return the product of x and the natural logarithm of y.
lmultiply_real_real = _XXX_TODO_XXX_("lmultiply")
lmultiply_int_real = _to_float_1(lmultiply_real_real)
lmultiply_real_int = _to_float_2(lmultiply_real_real)
lmultiply_int_int = _to_float_1_2(lmultiply_real_real)

# R lmultiply(T1 x, T2 y)
# Vectorized implementation of the lmultiply function
lmultiply_vectorized = _XXX_TODO_XXX_("lmultiply")
lmultiply_vector_vector = lmultiply_vectorized
lmultiply_rowvector_rowvector = lmultiply_vectorized
lmultiply_matrix_matrix = lmultiply_vectorized
lmultiply_array_array = lmultiply_vectorized

lmultiply_real_vector = lmultiply_vectorized
lmultiply_real_rowvector = lmultiply_vectorized
lmultiply_real_matrix = lmultiply_vectorized
lmultiply_real_array = lmultiply_vectorized
lmultiply_vector_real = lmultiply_vectorized
lmultiply_rowvector_real = lmultiply_vectorized
lmultiply_matrix_real = lmultiply_vectorized
lmultiply_array_real = lmultiply_vectorized

lmultiply_int_vector = _to_float_1(lmultiply_vectorized)
lmultiply_int_rowvector = _to_float_1(lmultiply_vectorized)
lmultiply_int_matrix = _to_float_1(lmultiply_vectorized)
lmultiply_int_array = _to_float_1(lmultiply_vectorized)
lmultiply_vector_int = _to_float_2(lmultiply_vectorized)
lmultiply_rowvector_int = _to_float_2(lmultiply_vectorized)
lmultiply_matrix_int = _to_float_2(lmultiply_vectorized)
lmultiply_array_int = _to_float_2(lmultiply_vectorized)

# R log1p(T x)
# natural logarithm of 1 plus x
log1p_int = _XXX_TODO_XXX_("log1p")
log1p_real = _XXX_TODO_XXX_("log1p")
log1p_vector = _XXX_TODO_XXX_("log1p")
log1p_rowvector = _XXX_TODO_XXX_("log1p")
log1p_matrix = _XXX_TODO_XXX_("log1p")
log1p_array = _XXX_TODO_XXX_("log1p")

# R log1m(T x)
# natural logarithm of 1 minus x
log1m_int = _XXX_TODO_XXX_("log1m")
log1m_real = _XXX_TODO_XXX_("log1m")
log1m_vector = _XXX_TODO_XXX_("log1m")
log1m_rowvector = _XXX_TODO_XXX_("log1m")
log1m_matrix = _XXX_TODO_XXX_("log1m")
log1m_array = _XXX_TODO_XXX_("log1m")

# R log1p_exp(T x)
# natural logarithm of one plus the natural exponentiation of x
from jax.nn import softplus

log1p_exp_int = lambda x: softplus(array(x, dtype=dtype_float))
log1p_exp_real = softplus
log1p_exp_vector = softplus
log1p_exp_rowvector = softplus
log1p_exp_matrix = softplus
log1p_exp_array = softplus

# R log1m_exp(T x)
# logarithm of one minus the natural exponentiation of x
log1m_exp_int = _XXX_TODO_XXX_("log1m_exp")
log1m_exp_real = _XXX_TODO_XXX_("log1m_exp")
log1m_exp_vector = _XXX_TODO_XXX_("log1m_exp")
log1m_exp_rowvector = _XXX_TODO_XXX_("log1m_exp")
log1m_exp_matrix = _XXX_TODO_XXX_("log1m_exp")
log1m_exp_array = _XXX_TODO_XXX_("log1m_exp")

# real log_diff_exp(real x, real y)
# Return the natural logarithm of the difference of the natural exponentiation of x and the natural exponentiation of y.
log_diff_exp_real_real = _XXX_TODO_XXX_("log_diff_exp")
log_diff_exp_int_real = _to_float_1(log_diff_exp_real_real)
log_diff_exp_real_int = _to_float_2(log_diff_exp_real_real)
log_diff_exp_int_int = _to_float_1_2(log_diff_exp_real_real)

# R log_diff_exp(T1 x, T2 y)
# Vectorized implementation of the log_diff_exp function
log_diff_exp_vectorized = _XXX_TODO_XXX_("log_diff_exp")
log_diff_exp_vector_vector = log_diff_exp_vectorized
log_diff_exp_rowvector_rowvector = log_diff_exp_vectorized
log_diff_exp_matrix_matrix = log_diff_exp_vectorized
log_diff_exp_array_array = log_diff_exp_vectorized

log_diff_exp_real_vector = log_diff_exp_vectorized
log_diff_exp_real_rowvector = log_diff_exp_vectorized
log_diff_exp_real_matrix = log_diff_exp_vectorized
log_diff_exp_real_array = log_diff_exp_vectorized
log_diff_exp_vector_real = log_diff_exp_vectorized
log_diff_exp_rowvector_real = log_diff_exp_vectorized
log_diff_exp_matrix_real = log_diff_exp_vectorized
log_diff_exp_array_real = log_diff_exp_vectorized

log_diff_exp_int_vector = _to_float_1(log_diff_exp_vectorized)
log_diff_exp_int_rowvector = _to_float_1(log_diff_exp_vectorized)
log_diff_exp_int_matrix = _to_float_1(log_diff_exp_vectorized)
log_diff_exp_int_array = _to_float_1(log_diff_exp_vectorized)
log_diff_exp_vector_int = _to_float_2(log_diff_exp_vectorized)
log_diff_exp_rowvector_int = _to_float_2(log_diff_exp_vectorized)
log_diff_exp_matrix_int = _to_float_2(log_diff_exp_vectorized)
log_diff_exp_array_int = _to_float_2(log_diff_exp_vectorized)

# real log_mix(real theta, real lp1, real lp2)
# Return the log mixture of the log densities lp1 and lp2 with mixing proportion theta, defined by log_mix(θ,λ1,λ2)=log(θexp(λ1)+(1−θ)exp(λ2))=log_sum_exp(log(θ)+λ1, log(1−θ)+λ2).
def log_mix_real_real_real(theta, lp1, lp2):
    return log_sum_exp_real_real(log_real(theta) + lp1, log_real(1 - theta) + lp2)
# XXX TODO: lifting to other types XXX

# real log_sum_exp(real x, real y)
# Return the natural logarithm of the sum of the natural exponentiation of x and the natural exponentiation of y. log_sum_exp(x,y)=log(exp(x)+exp(y))
from jax.numpy import logaddexp

log_sum_exp_real_real = logaddexp

# R log_inv_logit(T x)
# natural logarithm of the inverse logit function of x
from jax.nn import log_sigmoid

log_inv_logit_int = lambda x: log_sigmoid(array(x, dtype=dtype_float))
log_inv_logit_real = log_sigmoid
log_inv_logit_vector = log_sigmoid
log_inv_logit_rowvector = log_sigmoid
log_inv_logit_matrix = log_sigmoid
log_inv_logit_array = log_sigmoid

# R log1m_inv_logit(T x)
# natural logarithm of 1 minus the inverse logit function of x
log1m_inv_logit_int = lambda x: tlog(1 - inv_logit_int(x))
log1m_inv_logit_real = lambda x: tlog(1 - inv_logit_real(x))
log1m_inv_logit_vector = lambda x: tlog(1 - inv_logit_vector(x))
log1m_inv_logit_rowvector = lambda x: tlog(1 - inv_logit_rowvector(x))
log1m_inv_logit_matrix = lambda x: tlog(1 - inv_logit_matrix(x))
log1m_inv_logit_array = lambda x: tlog(1 - inv_logit_array(x))


## 4 Array Operations

## 4.1 Reductions

# 4.1.1 Minimum and Maximum

# real min(real[] x)
# The minimum value in x, or +∞  if x is size 0.
# int min(int[] x)
# The minimum value in x, or error if x is size 0.
from jax.numpy import min as tmin

min_array = tmin

# real max(real[] x)
# The maximum value in x, or −∞ if x is size 0.
# int max(int[] x)
# The maximum value in x, or error if x is size 0.
from jax.numpy import max as tmax

max_array = tmax


## 4.1.2 Sum, Product, and Log Sum of Exp

# int sum(int[] x)
# The sum of the elements in x, defined for x
# of size N by sum(x)
# real sum(real[] x)
# The sum of the elements in x; see definition above.
from jax.numpy import sum

sum_array = sum

# real prod(real[] x)
# The product of the elements in x, or 1 if x is size 0.
# real prod(int[] x)
# The product of the elements in x, product(x)={∏Nn=1xnifN>01ifN=0
from jax.numpy import prod as tprod

prod_array = tprod

# real log_sum_exp(real[] x)
# The natural logarithm of the sum of the exponentials of the elements in x, or −∞
# if the array is empty.
from jax.scipy.special import logsumexp

log_sum_exp_array = lambda x: logsumexp(x, 0)

## 4.1.3 Sample Mean, Variance, and Standard Deviation

# real mean(real[] x)
# The sample mean of the elements in x.
# It is an error to the call the mean function with an array of size 0.
from jax.numpy import mean

mean_array = mean

# real variance(real[] x)
# The sample variance of the elements in x.
# It is an error to call the variance function with an array of size 0.
from jax.numpy import var

variance_array = var

# real sd(real[] x)
# The sample standard deviation of elements in x.
# It is an error to call the sd function with an array of size 0.
from jax.numpy import std

sd_array = std


## 4.1.4 Euclidean Distance and Squared Distance

# real distance(vector x, vector y)
# real distance(vector x, row_vector y)
# real distance(row_vector x, vector y)
# real distance(row_vector x, row_vector y)
# The Euclidean distance between x and y
distance_vector_vector = _XXX_TODO_XXX_("distance")
distance_vector_rowvector = _XXX_TODO_XXX_("distance")
distance_rowvector_vector = _XXX_TODO_XXX_("distance")
distance_rowvector_rowvector = _XXX_TODO_XXX_("distance")

# real squared_distance(vector x, vector y)
# real squared_distance(vector x, row_vector [] y)
# real squared_distance(row_vector x, vector [] y)
# real squared_distance(row_vector x, row_vector[] y)
# The squared Euclidean distance between x and y
squared_distance_vector_vector = _XXX_TODO_XXX_("squared_distance")
squared_distance_vector_rowvector = _XXX_TODO_XXX_("squared_distance")
squared_distance_rowvector_vector = _XXX_TODO_XXX_("squared_distance")
squared_distance_rowvector_rowvector = _XXX_TODO_XXX_("squared_distance")


## 4.2 Array Size and Dimension Function

# int[] dims(T x)
# Return an integer array containing the dimensions of x; the type
# of the argument T can be any Stan type with up to 8 array
# dimensions.
dims_int = lambda x: array([], dtype=dtype_long)
dims_real = lambda x: array(x.shape)
dims_vector = lambda x: array(x.shape)
dims_rowvector = lambda x: array(x.shape)
dims_matrix = lambda x: array(x.shape)
dims_array = lambda x: array(x.shape)

# int num_elements(T[] x)
# Return the total number of elements in the array x including all
# elements in contained arrays, vectors, and matrices. T can be any
# array type. For example, if x is of type real[4,3] then
# num_elements(x) is 12, and if y is declared as matrix[3,4] y[5],
# then size(y) evaluates to 60.
num_elements_array = lambda x: tprod(array(x.shape))

# int size(T[] x)
# Return the number of elements in the array x; the type of the array T
# can be any type, but the size is just the size of the top level
# array, not the total number of elements contained. For example, if
# x is of type real[4,3] then size(x) is 4.
size_array = lambda x: x.shape[0]

## 4.3 Array Broadcasting

# T[] rep_array(T x, int n)
# Return the n array with every entry assigned to x.
rep_array_int_int = lambda x, n: x * ones(n, dtype=dtype_long)
rep_array_real_int = lambda x, n: x * ones(n, dtype=dtype_float)

# T[,] rep_array(T x, int m, int n)
# Return the m by n array with every entry assigned to x.
rep_array_int_int_int = lambda x, n, m: x * ones([n, m], dtype=dtype_long)
rep_array_real_int_int = lambda x, n, m: x * ones([n, m], dtype=dtype_float)

# T[,,] rep_array(T x, int k, int m, int n)
# Return the k by m by n array with every entry assigned to x.
rep_array_int_int_int_int = lambda x, k, n, m: x * ones([k, n, m], dtype=dtype_long)
rep_array_real_int_int_int = lambda x, k, n, m: x * ones([k, n, m], dtype=dtype_float)

## 4.4 Array concatenation

# T append_array(T x, T y)
# Return the concatenation of two arrays in the order of the arguments. T must be an N-dimensional array of any Stan type (with a maximum N of 7). All dimensions but the first must match.
append_array_array_array = _XXX_TODO_XXX_("append_array")

## 4.5 Sorting functions

# real[] sort_asc(real[] v)
# int[] sort_asc(int[] v)
# Sort the elements of v in ascending order
sort_asc_array = _XXX_TODO_XXX_("sort_asc")

# real[] sort_desc(real[] v)
# int[] sort_desc(int[] v)
# Sort the elements of v in descending order
sort_desc_array = _XXX_TODO_XXX_("sort_desc")

# int[] sort_indices_asc(real[] v)
# int[] sort_indices_asc(int[] v)
# Return an array of indices between 1 and the size of v, sorted to index v in ascending order.
sort_indices_asc_array = _XXX_TODO_XXX_("sort_indices_asc")

# int[] sort_indices_desc(real[] v)
# int[] sort_indices_desc(int[] v)
# Return an array of indices between 1 and the size of v, sorted to index v in descending order.
sort_indices_desc_array = _XXX_TODO_XXX_("sort_indices_desc")

# int rank(real[] v, int s)
# int rank(int[] v, int s)
# Number of components of v less than v[s]
rank_array = _XXX_TODO_XXX_("rank")

# 4.6 Reversing functions

# T[] reverse(T[] v)
# Return a new array containing the elements of the argument in reverse order.
reverse_array = _XXX_TODO_XXX_("reverse")


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

## 5.2 Matrix arithmetic operators

## 5.2.1 Negation prefix operators

# vector operator-(vector x)
# The negation of the vector x.
# This function is directly supported by the compiler

# row_vector operator-(row_vector x)
# The negation of the row vector x.
# This function is directly supported by the compiler

# matrix operator-(matrix x)
# The negation of the matrix x.
# This function is directly supported by the compiler

## 5.2.2 Infix matrix operators

# vector operator+(vector x, vector y)
# The sum of the vectors x and y.
# This function is directly supported by the compiler

# row_vector operator+(row_vector x, row_vector y)
# The sum of the row vectors x and y.
# This function is directly supported by the compiler

# matrix operator+(matrix x, matrix y)
# The sum of the matrices x and y
# This function is directly supported by the compiler

# vector operator-(vector x, vector y)
# The difference between the vectors x and y.
# This function is directly supported by the compiler

# row_vector operator-(row_vector x, row_vector y)
# The difference between the row vectors x and y
# This function is directly supported by the compiler

# matrix operator-(matrix x, matrix y)
# The difference between the matrices x and y
# This function is directly supported by the compiler

# vector operator*(real x, vector y)
# The product of the scalar x and vector y
# This function is directly supported by the compiler

# row_vector operator*(real x, row_vector y)
# The product of the scalar x and the row vector y
# This function is directly supported by the compiler

# matrix operator*(real x, matrix y)
# The product of the scalar x and the matrix y
# This function is directly supported by the compiler

# vector operator*(vector x, real y)
# The product of the scalar y and vector x
# This function is directly supported by the compiler

# matrix operator*(vector x, row_vector y)
# The product of the vector x and row vector y
# This function is directly supported by the compiler

# row_vector operator*(row_vector x, real y)
# The product of the scalar y and row vector x
# This function is directly supported by the compiler

# real operator*(row_vector x, vector y)
# The product of the row vector x and vector y
# This function is directly supported by the compiler

# row_vector operator*(row_vector x, matrix y)
# The product of the row vector x and matrix y
# This function is directly supported by the compiler

# matrix operator*(matrix x, real y)
# The product of the scalar y and matrix x
# This function is directly supported by the compiler

# vector operator*(matrix x, vector y)
# The product of the matrix x and vector y
# This function is directly supported by the compiler

# matrix operator*(matrix x, matrix y)
# The product of the matrices x and y
# This function is directly supported by the compiler

## 5.2.3 Broadcast infix operators

# vector operator+(vector x, real y)
# The result of adding y to every entry in the vector x
# This function is directly supported by the compiler

# vector operator+(real x, vector y)
# The result of adding x to every entry in the vector y
# This function is directly supported by the compiler

# row_vector operator+(row_vector x, real y)
# The result of adding y to every entry in the row vector x
# This function is directly supported by the compiler

# row_vector operator+(real x, row_vector y)
# The result of adding x to every entry in the row vector y
# This function is directly supported by the compiler

# matrix operator+(matrix x, real y)
# The result of adding y to every entry in the matrix x
# This function is directly supported by the compiler

# matrix operator+(real x, matrix y)
# The result of adding x to every entry in the matrix y
# This function is directly supported by the compiler

# vector operator-(vector x, real y)
# The result of subtracting y from every entry in the vector x
# This function is directly supported by the compiler

# vector operator-(real x, vector y)
# The result of adding x to every entry in the negation of the vector y
# This function is directly supported by the compiler

# row_vector operator-(row_vector x, real y)
# The result of subtracting y from every entry in the row vector x
# This function is directly supported by the compiler

# row_vector operator-(real x, row_vector y)
# The result of adding x to every entry in the negation of the row vector y
# This function is directly supported by the compiler

# matrix operator-(matrix x, real y)
# The result of subtracting y from every entry in the matrix x
# This function is directly supported by the compiler

# matrix operator-(real x, matrix y)
# The result of adding x to every entry in negation of the matrix y
# This function is directly supported by the compiler

# vector operator/(vector x, real y)
# The result of dividing each entry in the vector x by y
# This function is directly supported by the compiler

# row_vector operator/(row_vector x, real y)
# The result of dividing each entry in the row vector x by y
# This function is directly supported by the compiler

# matrix operator/(matrix x, real y)
# The result of dividing each entry in the matrix x by y
# This function is directly supported by the compiler

## 5.3 Transposition operator

# matrix operator'(matrix x)
# The transpose of the matrix x, written as x'
# This function is directly supported by the compiler

# row_vector operator'(vector x)
# The transpose of the vector x, written as x'
# This function is directly supported by the compiler

# vector operator'(row_vector x)
# The transpose of the row vector x, written as x'
# This function is directly supported by the compiler

## 5.4 Elementwise functions

# vector operator.*(vector x, vector y)
# The elementwise product of y and x
# This function is directly supported by the compiler

# row_vector operator.*(row_vector x, row_vector y)
# The elementwise product of y and x
# This function is directly supported by the compiler

# matrix operator.*(matrix x, matrix y)
# The elementwise product of y and x
# This function is directly supported by the compiler

# vector operator./(vector x, vector y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# vector operator./(vector x, real y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# vector operator./(real x, vector y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# row_vector operator./(row_vector x, row_vector y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# row_vector operator./(row_vector x, real y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# row_vector operator./(real x, row_vector y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# matrix operator./(matrix x, matrix y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# matrix operator./(matrix x, real y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# matrix operator./(real x, matrix y)
# The elementwise quotient of y and x
# This function is directly supported by the compiler

# vector operator.^(vector x, vector y)
# The elementwise power of y and x
# This function is directly supported by the compiler

# vector operator.^(vector x, real y)
# The elementwise power of y and x
# This function is directly supported by the compiler

# vector operator.^(real x, vector y)
# The elementwise power of y and x
# This function is directly supported by the compiler

# row_vector operator.^(row_vector x, row_vector y)
# The elementwise power of y and x
# This function is directly supported by the compiler

# row_vector operator.^(row_vector x, real y)
# The elementwise power of y and x
# This function is directly supported by the compiler

# row_vector operator.^(real x, row_vector y)
# The elementwise power of y and x
# This function is directly supported by the compiler

# matrix operator.^(matrix x, matrix y)
# The elementwise power of y and x
# This function is directly supported by the compiler

# matrix operator.^(matrix x, real y)
# The elementwise power of y and x
# This function is directly supported by the compiler

# matrix operator.^(real x, matrix y)
# The elementwise power of y and x
# This function is directly supported by the compiler

## 5.5 Dot Products and Specialized Products

# real dot_product(vector x, vector y)
# The dot product of x and y
from jax.numpy import dot as tdot

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
columns_dot_product_vector_vector = _XXX_TODO_XXX_("columns_dot_product")

# row_vector columns_dot_product(row_vector x, row_vector y)
# The dot product of the columns of x and y
columns_dot_product_rowvector_rowvector = _XXX_TODO_XXX_("columns_dot_product")

# row_vector columns_dot_product(matrix x, matrix y)
# The dot product of the columns of x and y
columns_dot_product_matrix_matrix = _XXX_TODO_XXX_("columns_dot_product")

# vector rows_dot_product(vector x, vector y)
# The dot product of the rows of x and y
rows_dot_product_vector_vector = _XXX_TODO_XXX_("rows_dot_product")

# vector rows_dot_product(row_vector x, row_vector y)
# The dot product of the rows of x and y
rows_dot_product_rowvector_rowvector = _XXX_TODO_XXX_("rows_dot_product")

# vector rows_dot_product(matrix x, matrix y)
# The dot product of the rows of x and y
rows_dot_product_matrix_matrix = _XXX_TODO_XXX_("rows_dot_product")

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
tcrossprod_matrix = _XXX_TODO_XXX_("tcrossprod")

# matrix crossprod(matrix x)
# The product of x premultiplied by its own transpose, similar to the crossprod(x) function in R. The result is a symmetric matrix.
crossprod_matrix = _XXX_TODO_XXX_("crossprod")

# matrix quad_form(matrix A, matrix B)
# The quadratic form, i.e., B' * A * B.
quad_form_matrix_matrix = _XXX_TODO_XXX_("quad_form")

# real quad_form(matrix A, vector B)
# The quadratic form, i.e., B' * A * B.
quad_form_matrix_vector = _XXX_TODO_XXX_("quad_form")

# matrix quad_form_diag(matrix m, vector v)
# The quadratic form using the column vector v as a diagonal matrix, i.e., diag_matrix(v) * m * diag_matrix(v).
quad_form_diag_matrix_vector = _XXX_TODO_XXX_("quad_form_diag")

# matrix quad_form_diag(matrix m, row_vector rv)
# The quadratic form using the row vector rv as a diagonal matrix, i.e., diag_matrix(rv) * m * diag_matrix(rv).
quad_form_diag_matrix_row_vector = _XXX_TODO_XXX_("quad_form_diag")

# matrix quad_form_sym(matrix A, matrix B)
# Similarly to quad_form, gives B' * A * B, but additionally checks if A is symmetric and ensures that the result is also symmetric.
quad_form_sym_matrix_matrix = _XXX_TODO_XXX_("quad_form_sym")

# real quad_form_sym(matrix A, vector B)
# Similarly to quad_form, gives B' * A * B, but additionally checks if A is symmetric and ensures that the result is also symmetric.
quad_form_sym_matrix_vector = _XXX_TODO_XXX_("quad_form_sym")

# real trace_quad_form(matrix A, matrix B)
# The trace of the quadratic form, i.e., trace(B' * A * B).
trace_quad_form_matrix_matrix = _XXX_TODO_XXX_("trace_quad_form")

# real trace_gen_quad_form(matrix D, matrix A, matrix B)
# The trace of a generalized quadratic form, i.e., trace(D * B' * A * B).
trace_gen_quad_form_matrix_matrix_matrix = _XXX_TODO_XXX_("trace_gen_quad_form")

# matrix multiply_lower_tri_self_transpose(matrix x)
# The product of the lower triangular portion of x (including the diagonal) times its own transpose; that is, if L is a matrix of the same dimensions as x with L(m,n) equal to x(m,n) for n≤m
# and L(m,n) equal to 0 if n>m, the result is the symmetric matrix LL⊤. This is a specialization of tcrossprod(x) for lower-triangular matrices. The input matrix does not need to be square.
multiply_lower_tri_self_matrix = _XXX_TODO_XXX_("multiply_lower_tri_self")

# matrix diag_pre_multiply(vector v, matrix m)
# Return the product of the diagonal matrix formed from the vector v and the matrix m, i.e., diag_matrix(v) * m.
diag_pre_multiply_vector_matrix = _XXX_TODO_XXX_("diag_pre_multiply")

# matrix diag_pre_multiply(row_vector rv, matrix m)
# Return the product of the diagonal matrix formed from the vector rv and the matrix m, i.e., diag_matrix(rv) * m.
diag_pre_multiply_rowvector_matrix = _XXX_TODO_XXX_("diag_pre_multiply")

# matrix diag_post_multiply(matrix m, vector v)
# Return the product of the matrix m and the diagonal matrix formed from the vector v, i.e., m * diag_matrix(v).
diag_post_multiply_matrix_vector = _XXX_TODO_XXX_("diag_post_multiply")

# matrix diag_post_multiply(matrix m, row_vector rv)
# Return the product of the matrix m and the diagonal matrix formed from the the row vector rv, i.e., m * diag_matrix(rv).
diag_post_multiply_matrix_rowvector = _XXX_TODO_XXX_("diag_post_multiply")

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
prod_vector = tprod

# real prod(row_vector x)
# The product of the values in x, or 1 if x is empty
prod_rowvector = tprod

# real prod(matrix x)
# The product of the values in x, or 1 if x is empty
prod_matrix = tprod

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
from jax.numpy import transpose, tile

rep_matrix_vector_int = lambda v, n: tile(transpose(array([v])), (1, n))

# matrix rep_matrix(row_vector rv, int m)
# Return the m by n matrix consisting of m copies of the row vector rv of size n.
rep_matrix_rowvector_int = lambda v, n: tile(array([v]), (n, 1))

## 5.8 Diagonal Matrix Functions
from jax.numpy import diag, eye

# matrix add_diag(matrix m, row_vector d)
# Add row_vector d to the diagonal of matrix m.

add_diag_matrix_rowvector = lambda m, rv: m + diag(rv)

# matrix add_diag(matrix m, vector d)
# Add vector d to the diagonal of matrix m.

add_diag_matrix_vector = lambda m, v: m + diag(v)

# matrix add_diag(matrix m, real d)
# Add scalar d to every diagonal element of matrix m.

add_diag_matrix_real = lambda m, r: r * eye(m.shape[0]) + m

# vector diagonal(matrix x)
# The diagonal of the matrix x

diagonal_matrix = lambda m: diag(m)

# matrix diag_matrix(vector x)
# The diagonal matrix with diagonal x

diag_matrix_vector = lambda v: diag(v)

# matrix identity_matrix(int k)
# Create an identity matrix of size k×k

identity_matrix_int = _XXX_TODO_XXX_("identity_matrix")

# 5.9 Container construction functions

# real[] linspaced_array(int n, data real lower, data real upper)
# Create a real array of length n of equidistantly-spaced elements between lower and upper
linspaced_array_int_real_real = _XXX_TODO_XXX_("linspaced_array")
linspaced_array_int_int_real = _to_float_1(linspaced_array_int_real_real)
linspaced_array_int_real_int = _to_float_2(linspaced_array_int_real_real)
linspaced_array_int_int_int = _to_float_1_2(linspaced_array_int_real_real)

# real[] linspaced_int_array(int n, int lower, int upper)
# Create a regularly spaced, increasing integer array of length n between lower and upper, inclusively. If (upper - lower) / (n - 1) is less than one, repeat each output (n - 1) / (upper - lower) times. If neither (upper - lower) / (n - 1) or (n - 1) / (upper - lower) are integers, upper is reduced until one of these is true.
linspaced_int_array_int_int_int = _XXX_TODO_XXX_("linspaced_int_array")

# vector linspaced_vector(int n, data real lower, data real upper)
# Create an n-dimensional vector of equidistantly-spaced elements between lower and upper
linspaced_vector_int_real_real = _XXX_TODO_XXX_("linspaced_vector")
linspaced_vector_int_int_real = _to_float_1(linspaced_vector_int_real_real)
linspaced_vector_int_real_int = _to_float_2(linspaced_vector_int_real_real)
linspaced_vector_int_int_int = _to_float_1_2(linspaced_vector_int_real_real)

# row_vector linspaced_row_vector(int n, data real lower, data real upper)
# Create an n-dimensional row-vector of equidistantly-spaced elements between lower and upper
linspaced_row_vector_int_real_real = _XXX_TODO_XXX_("linspaced_row_vector")
linspaced_row_vector_int_int_real = _to_float_1(linspaced_row_vector_int_real_real)
linspaced_row_vector_int_real_int = _to_float_2(linspaced_row_vector_int_real_real)
linspaced_row_vector_int_int_int = _to_float_1_2(linspaced_row_vector_int_real_real)

# int[] one_hot_int_array(int n, int k)
# Create a one-hot encoded int array of length n with array[k] = 1
one_hot_int_array_int_int = _XXX_TODO_XXX_("one_hot_int_array")

# real[] one_hot_array(int n, int k)
# Create a one-hot encoded real array of length n with array[k] = 1
one_hot_array_int_int = _XXX_TODO_XXX_("one_hot_array")

# vector one_hot_vector(int n, int k)
# Create an n-dimensional one-hot encoded vector with vector[k] = 1
one_hot_vector_int_int = _XXX_TODO_XXX_("one_hot_vector")

# row_vector one_hot_row_vector(int n, int k)
# Create an n-dimensional one-hot encoded row-vector with row_vector[k] = 1
one_hot_row_vector_int_int = _XXX_TODO_XXX_("one_hot_row_vector")

# int[] ones_int_array(int n)
# Create an int array of length n of all ones
ones_int_array_int = _XXX_TODO_XXX_("ones_int_array")

# real[] ones_array(int n)
# Create a real array of length n of all ones
ones_array_int = _XXX_TODO_XXX_("ones_array")

# vector ones_vector(int n)
# Create an n-dimensional vector of all ones
ones_vector_int = _XXX_TODO_XXX_("ones_vector")

# row_vector ones_row_vector(int n)
# Create an n-dimensional row-vector of all ones
ones_row_vector_int = _XXX_TODO_XXX_("ones_row_vector")

# int[] zeros_int_array(int n)
# Create an int array of length n of all zeros
zeros_int_array_int = _XXX_TODO_XXX_("zeros_int_array")

# real[] zeros_array(int n)
# Create a real array of length n of all zeros
zeros_array_int = _XXX_TODO_XXX_("zeros_array")

# vector zeros_vector(int n)
# Create an n-dimensional vector of all zeros
zeros_vector_int = _XXX_TODO_XXX_("zeros_vector")

# row_vector zeros_row_vector(int n)
# Create an n-dimensional row-vector of all zeros
zeros_row_vector_int = _XXX_TODO_XXX_("zeros_row_vector")

# vector uniform_simplex(int n)
# Create an n-dimensional simplex with elements vector[i] = 1 / n for all i∈1,…,n
uniform_simplex_int = _XXX_TODO_XXX_("uniform_simplex")


## 5.10 Slicing and Blocking Functions

## 5.10.1 Columns and Rows

# vector col(matrix x, int n)
# The n-th column of matrix x
col_matrix_int = lambda x, n: x[:, n - 1]

# row_vector row(matrix x, int m)
# The m-th row of matrix x
row_matrix_int = lambda x, m: x[m - 1]

## 5.10.2 Block Operations

## 5.10.2.1 Matrix Slicing Operations

# matrix block(matrix x, int i, int j, int n_rows, int n_cols)
# Return the submatrix of x that starts at row i and column j and extends n_rows rows and n_cols columns.
block_matrix_int_int_int_int = lambda x, i, j, n_rows, n_cols: x[
    i - 1 : i - 1 + n_rows, j - 1 : j - 1 + n_cols
]

# vector sub_col(matrix x, int i, int j, int n_rows)
# Return the sub-column of x that starts at row i and column j and extends n_rows rows and 1 column.
sub_col_matrix_int_int_int = lambda x, i, j, n_rows: (
    x[i - 1 : i - 1 + n_rows, j - 1 : j]
)[:, 0]

# row_vector sub_row(matrix x, int i, int j, int n_cols)
# Return the sub-row of x that starts at row i and column j and extends 1 row and n_cols columns.
sub_row_matrix_int_int_int = lambda x, i, y, n_cols: x[
    i - 1 : i, j - 1 : j - 1 + n_cols
]

# 5.10.2.2 Vector and Array Slicing Operations

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


## 5.11 Matrix Concatenation

## 5.11.0.1 Horizontal concatenation

from jax.numpy import append

# matrix append_col(matrix x, matrix y)
# Combine matrices x and y by columns. The matrices must have the same number of rows.
append_col_matrix_matrix = lambda x, y: append(x, y, axis=1)

# matrix append_col(matrix x, vector y)
# Combine matrix x and vector y by columns. The matrix and the vector must have the same number of rows.
append_col_matrix_vector = lambda x, y: append(x, transpose(array([y])), axis=1)

# matrix append_col(vector x, matrix y)
# Combine vector x and matrix y by columns. The vector and the matrix must have the same number of rows.
append_col_vector_matrix = lambda x, y: append(transpose(array([x])), y, axis=1)

# matrix append_col(vector x, vector y)
# Combine vectors x and y by columns. The vectors must have the same number of rows.
append_col_vector_vector = lambda x, y: append(
    transpose(array([x])), transpose(array([y])), axis=1
)

# row_vector append_col(row_vector x, row_vector y)
# Combine row vectors x and y of any size into another row vector.
append_col_rowvector_rowvector = lambda x, y: append(x, y, axis=0)

# row_vector append_col(real x, row_vector y)
# Append x to the front of y, returning another row vector.
append_col_real_rowvector = lambda x, y: append(array([x], dtype=dtype_float), y)
append_col_int_rowvector = lambda x, y: append(array([x], dtype=dtype_float), y)

# row_vector append_col(row_vector x, real y)
# Append y to the end of x, returning another row vector.
append_col_rowvector_real = lambda x, y: append(x, array([y], dtype=dtype_float))
append_col_rowvector_int = lambda x, y: append(x, array([y], dtype=dtype_float))

## 5.11.0.2 Vertical concatenation

# matrix append_row(matrix x, matrix y)
# Combine matrices x and y by rows. The matrices must have the same number of columns.
append_row_matrix_matrix = lambda x, y: append(x, y, axis=0)

# matrix append_row(matrix x, row_vector y)
# Combine matrix x and row vector y by rows. The matrix and the row vector must have the same number of columns.
append_row_matrix_rowvector = lambda x, y: append(x, array([y]), axis=0)

# matrix append_row(row_vector x, matrix y)
# Combine row vector x and matrix y by rows. The row vector and the matrix must have the same number of columns.
append_row_rowvector_matrix = lambda x, y: append(array([x]), y, axis=0)

# matrix append_row(row_vector x, row_vector y)
# Combine row vectors x and y by row. The row vectors must have the same number of columns.
append_row_rowvector_rowvector = lambda x, y: append(array([x]), array([y]), axis=0)

# vector append_row(vector x, vector y)
# Concatenate vectors x and y of any size into another vector.
append_row_vector_vector = lambda x, y: append(x, y)

# vector append_row(real x, vector y)
# Append x to the top of y, returning another vector.
append_row_real_vector = lambda x, y: append(array([x], dtype=dtype_float), y)
append_row_int_vector = lambda x, y: append(array([x], dtype=dtype_float), y)

# vector append_row(vector x, real y)
# Append y to the bottom of x, returning another vector.
append_row_vector_real = lambda x, y: append(x, array([y], dtype=dtype_float))
append_row_vector_int = lambda x, y: append(x, array([y], dtype=dtype_float))

## 5.12 Special Matrix Functions
## 5.12.1 Softmax

# vector softmax(vector x)
# The softmax of x
from jax.nn import softmax as tsoftmax

softmax_vector = tsoftmax

# vector log_softmax(vector x)
# The natural logarithm of the softmax of x
from jax.nn import log_softmax as tlogsoftmax

softmax_vector = tlogsoftmax

# 5.12.2 Cumulative Sums

# real[] cumulative_sum(real[] x)
# The cumulative sum of x
from jax.numpy import cumsum as tcumsum

cumulative_sum_array = tcumsum

# vector cumulative_sum(vector v)
# The cumulative sum of v
cumulative_sum_vector = tcumsum

# row_vector cumulative_sum(row_vector rv)
# The cumulative sum of rv
cumulative_sum_rowvector = tcumsum

## 5.13 Covariance Functions
## 5.13.1 Exponentiated quadratic covariance function

from jax.numpy import power as pow
from jax.numpy.linalg import norm

def cov_exp_quad(x, alpha, rho):
    return alpha * alpha * texp(-0.5 * pow(norm(x) / rho, 2))

# matrix cov_exp_quad(row_vectors x, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x.
cov_exp_quad_rowvector_real_real = cov_exp_quad

# matrix cov_exp_quad(vectors x, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x.
cov_exp_quad_vector_real_real = cov_exp_quad

# matrix cov_exp_quad(real[] x, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x.
cov_exp_quad_array_real_real = cov_exp_quad

# matrix cov_exp_quad(row_vectors x1, row_vectors x2, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x1 and x2.
cov_exp_quad_rowvector_rowvector_real_real = cov_exp_quad

# matrix cov_exp_quad(vectors x1, vectors x2, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x1 and x2.
cov_exp_quad_vector_vector_real_real = cov_exp_quad

# matrix cov_exp_quad(real[] x1, real[] x2, real alpha, real rho)
# The covariance matrix with an exponentiated quadratic kernel of x1 and x2.
cov_exp_quad_array_array_real_real = cov_exp_quad

## 5.14 Linear Algebra Functions and Solvers

## 5.14.1.1 Matrix division operators

# row_vector operator/(row_vector b, matrix A)
# The right division of b by A; equivalently b * inverse(A)
# This function is directly supported by the compiler

# matrix operator/(matrix B, matrix A)
# The right division of B by A; equivalently B * inverse(A)
# This function is directly supported by the compiler

# vector operator\(matrix A, vector b)
# The left division of A by b; equivalently inverse(A) * b
# This function is directly supported by the compiler

# matrix operator\(matrix A, matrix B)
# The left division of A by B; equivalently inverse(A) * B
# This function is directly supported by the compiler

## 5.14.1.2 Lower-triangular matrix division functions

# vector mdivide_left_tri_low(matrix A, vector b)
# The left division of b by a lower-triangular view of A; algebraically equivalent to the less efficient and stable form inverse(tri(A)) * b, where tri(A) is the lower-triangular portion of A with the above-diagonal entries set to zero.
mdivide_left_tri_low_matrix_vector = _XXX_TODO_XXX_("mdivide_left_tri_low")

# matrix mdivide_left_tri_low(matrix A, matrix B)
# The left division of B by a triangular view of A; algebraically equivalent to the less efficient and stable form inverse(tri(A)) * B, where tri(A) is the lower-triangular portion of A with the above-diagonal entries set to zero.
mdivide_left_tri_low_matrix_matrix = _XXX_TODO_XXX_("mdivide_left_tri_low")

# row_vector mdivide_right_tri_low(row_vector b, matrix A)
# The right division of b by a triangular view of A; algebraically equivalent to the less efficient and stable form b * inverse(tri(A)), where tri(A) is the lower-triangular portion of A with the above-diagonal entries set to zero.
mdivide_right_tri_low_row_vector_matrix = _XXX_TODO_XXX_("mdivide_right_tri_low")

# matrix mdivide_right_tri_low(matrix B, matrix A)
# The right division of B by a triangular view of A; algebraically equivalent to the less efficient and stable form B * inverse(tri(A)), where tri(A) is the lower-triangular portion of A with the above-diagonal entries set to zero.
mdivide_right_tri_low_matrix_matrix = _XXX_TODO_XXX_("mdivide_right_tri_low")

## 5.14.2 Symmetric positive-definite matrix division functions

# matrix mdivide_left_spd(matrix A, vector b)
# The left division of b by the symmetric, positive-definite matrix A; algebraically equivalent to the less efficient and stable form inverse(A) * b.
mdivide_left_spd_matrix_vector = _XXX_TODO_XXX_("mdivide_left_spd")

# vector mdivide_left_spd(matrix A, matrix B)
# The left division of B by the symmetric, positive-definite matrix A; algebraically equivalent to the less efficient and stable form inverse(A) * B.
mdivide_left_spd_matrix_matrix = _XXX_TODO_XXX_("mdivide_left_spd")

# row_vector mdivide_right_spd(row_vector b, matrix A)
# The right division of b by the symmetric, positive-definite matrix A; algebraically equivalent to the less efficient and stable form b * inverse(A).
mdivide_right_spd_row_vector_matrix = _XXX_TODO_XXX_("mdivide_right_spd")

# matrix mdivide_right_spd(matrix B, matrix A)
# The right division of B by the symmetric, positive-definite matrix A; algebraically equivalent to the less efficient and stable form B * inverse(A).
mdivide_right_spd_matrix_matrix = _XXX_TODO_XXX_("mdivide_right_spd")

## 5.14.3 Matrix exponential

# matrix matrix_exp(matrix A)
# The matrix exponential of A
matrix_exp_matrix = _XXX_TODO_XXX_("matrix_exp")

# matrix matrix_exp_multiply(matrix A, matrix B)
# The multiplication of matrix exponential of A and matrix B; algebraically equivalent to the less efficient form matrix_exp(A) * B.
matrix_exp_multiply_matrix_matrix = _XXX_TODO_XXX_("matrix_exp_multiply")

# matrix scale_matrix_exp_multiply(real t, matrix A, matrix B)
# The multiplication of matrix exponential of tA and matrix B; algebraically equivalent to the less efficient form matrix_exp(t * A) * B.
scale_matrix_exp_multiply_real_matrix_matrix = _XXX_TODO_XXX_("scale_matrix_exp_multiply")
scale_matrix_exp_multiply_int_matrix_matrix = _to_float_1(scale_matrix_exp_multiply_real_matrix_matrix)

## 5.14.4 Matrix power

# matrix matrix_power(matrix A, int B)
# Matrix A raised to the power B.
matrix_power_matrix_int = _XXX_TODO_XXX_("matrix_power")

## 5.14.5 Linear algebra functions

## 5.14.5.1 Trace

# real trace(matrix A)
# The trace of A, or 0 if A is empty; A is not required to be diagonal
trace_matrix = _XXX_TODO_XXX_("trace")

## 5.14.5.2 Determinants

# real determinant(matrix A)
# The determinant of A
determinant_matrix = _XXX_TODO_XXX_("determinant")

# real log_determinant(matrix A)
# The log of the absolute value of the determinant of A
log_determinant_matrix = _XXX_TODO_XXX_("log_determinant")

## 5.14.5.3 Inverses

# matrix inverse(matrix A)
# The inverse of A
inverse_matrix = _XXX_TODO_XXX_("inverse")

# matrix inverse_spd(matrix A)
# The inverse of A where A is symmetric, positive definite. This version is faster and more arithmetically stable when the input is symmetric and positive definite.
inverse_spd_matrix = _XXX_TODO_XXX_("inverse_spd")

## 5.14.5.4 Generalized Inverse

# matrix generalized_inverse(matrix A)
# The generalized inverse of A
generalized_inverse_matrix = _XXX_TODO_XXX_("generalized_inverse")

## 5.14.5.5 Eigendecomposition

# vector eigenvalues_sym(matrix A)
# The vector of eigenvalues of a symmetric matrix A in ascending order
eigenvalues_sym_matrix = _XXX_TODO_XXX_("eigenvalues_sym")

# matrix eigenvectors_sym(matrix A)
# The matrix with the (column) eigenvectors of symmetric matrix A in the same order as returned by the function eigenvalues_sym
eigenvectors_sym_matrix = _XXX_TODO_XXX_("eigenvectors_sym")

## 5.14.5.6 QR decomposition

# matrix qr_thin_Q(matrix A)
# The orthogonal matrix in the thin QR decomposition of A, which implies that the resulting matrix has the same dimensions as A
qr_thin_Q_matrix = _XXX_TODO_XXX_("qr_thin_Q")

# matrix qr_thin_R(matrix A)
# The upper triangular matrix in the thin QR decomposition of A, which implies that the resulting matrix is square with the same number of columns as A
qr_thin_R_matrix = _XXX_TODO_XXX_("qr_thin_R")

# matrix qr_Q(matrix A)
# The orthogonal matrix in the fat QR decomposition of A, which implies that the resulting matrix is square with the same number of rows as A
qr_Q_matrix = _XXX_TODO_XXX_("qr_Q")

# matrix qr_R(matrix A)
# The upper trapezoidal matrix in the fat QR decomposition of A, which implies that the resulting matrix will be rectangular with the same dimensions as A
qr_R_matrix = _XXX_TODO_XXX_("qr_R")

## 5.14.5.7 Cholesky decomposition

from jax.numpy.linalg import cholesky

# matrix cholesky_decompose(matrix A)
# The lower-triangular Cholesky factor of the symmetric positive-definite matrix A
cholesky_decompose_matrix = lambda m: cholesky(m)

## 5.14.5.8 Singular value decomposition

# vector singular_values(matrix A)
# The singular values of A in descending order
singular_values_matrix = _XXX_TODO_XXX_("singular_values")

# vector svd_U(matrix A)
# The left-singular vectors of A
svd_U_matrix = _XXX_TODO_XXX_("svd_U")

# vector svd_V(matrix A)
# The right-singular vectors of A
svd_V_matrix = _XXX_TODO_XXX_("svd_V")

## 5.15 Sort functions

# vector sort_asc(vector v)
# Sort the elements of v in ascending order
sort_asc_vector = _XXX_TODO_XXX_("sort_asc")

# row_vector sort_asc(row_vector v)
# Sort the elements of v in ascending order
sort_asc_row_vector = _XXX_TODO_XXX_("sort_asc")

# vector sort_desc(vector v)
# Sort the elements of v in descending order
sort_desc_vector = _XXX_TODO_XXX_("sort_desc")

# row_vector sort_desc(row_vector v)
# Sort the elements of v in descending order
sort_desc_row_vector_row_vector = _XXX_TODO_XXX_("sort_desc")

# int[] sort_indices_asc(vector v)
# Return an array of indices between 1 and the size of v, sorted to index v in ascending order.
sort_indices_asc_vector = _XXX_TODO_XXX_("sort_indices_asc")

# int[] sort_indices_asc(row_vector v)
# Return an array of indices between 1 and the size of v, sorted to index v in ascending order.
sort_indices_asc_row_vector = _XXX_TODO_XXX_("sort_indices_asc")

# int[] sort_indices_desc(vector v)
# Return an array of indices between 1 and the size of v, sorted to index v in descending order.
sort_indices_desc_vector = _XXX_TODO_XXX_("sort_indices_desc")

# int[] sort_indices_desc(row_vector v)
# Return an array of indices between 1 and the size of v, sorted to index v in descending order.
sort_indices_desc_row_vector = _XXX_TODO_XXX_("sort_indices_desc")

# int rank(vector v, int s)
# Number of components of v less than v[s]
rank_vector_int = _XXX_TODO_XXX_("rank")

# int rank(row_vector v, int s)
# Number of components of v less than v[s]
rank_row_vector_int = _XXX_TODO_XXX_("rank")

## 5.16 Reverse functions

# vector reverse(vector v)
# Return a new vector containing the elements of the argument in reverse order.
reverse_vector = _XXX_TODO_XXX_("reverse")

# row_vector reverse(row_vector v)
# Return a new row vector containing the elements of the argument in reverse order.
reverse_row_vector = _XXX_TODO_XXX_("reverse")

## 6 Sparse Matrix Operations

## 6.1 Compressed row storage

# no function definition

## 6.2 Conversion functions

## 6.2.1 Dense to sparse conversion

# vector csr_extract_w(matrix a)
# Return non-zero values in matrix a; see section compressed row storage.
csr_extract_w_matrix = _XXX_TODO_XXX_("csr_extract_w")

# int[] csr_extract_v(matrix a)
# Return column indices for values in csr_extract_w(a); see compressed row storage.
csr_extract_v_matrix = _XXX_TODO_XXX_("csr_extract_v")

# int[] csr_extract_u(matrix a)
# Return array of row starting indices for entries in csr_extract_w(a) followed by the size of csr_extract_w(a) plus one; see section compressed row storage.
csr_extract_u_matrix = _XXX_TODO_XXX_("csr_extract_u")

## 6.2.2 Sparse to dense conversion

# matrix csr_to_dense_matrix(int m, int n, vector w, int[] v, int[] u)
# Return dense m×n matrix with non-zero matrix entries w, column indices v, and row starting indices u; the vector w and array v must be the same size (corresponding to the total number of nonzero entries in the matrix), array v must have index values bounded by m, array u must have length equal to m + 1 and contain index values bounded by the number of nonzeros (except for the last entry, which must be equal to the number of nonzeros plus one). See section compressed row storage for more details.
csr_to_dense_matrix_int_int_vector_int_int = _XXX_TODO_XXX_("csr_to_dense_matrix")

## 6.3 Sparse matrix arithmetic

## 6.3.1 Sparse matrix multiplication

# vector csr_matrix_times_vector(int m, int n, vector w, int[] v, int[] u, vector b)
# Multiply the m×n matrix represented by values w, column indices v, and row start indices u by the vector b; see compressed row storage.
csr_matrix_times_vector_int_int_vector_int_int_vector = _XXX_TODO_XXX_("csr_matrix_times_vector")

## 7. Mixed Operations

# matrix to_matrix(matrix m)
# Return the matrix m itself.
to_matrix_matrix = lambda m: m

# matrix to_matrix(vector v)
# Convert the column vector v to a size(v) by 1 matrix.
to_matrix_vector = lambda v: transpose(array([v]))

# matrix to_matrix(row_vector v)
# Convert the row vector v to a 1 by size(v) matrix.
to_matrix_rowvector = lambda v: array([v])

# matrix to_matrix(matrix m, int m, int n)
# Convert a matrix m to a matrix with m rows and n columns filled in column-major order.
to_matrix_matrix_int_int = lambda mat, m, n: transpose(mat).reshape(m, n)

# matrix to_matrix(vector v, int m, int n)
# Convert a vector v to a matrix with m rows and n columns filled in column-major order.
to_matrix_vector_int_int = lambda v, m, n: transpose(v.reshape(n, m))

# matrix to_matrix(row_vector v, int m, int n)
# Convert a row_vector a to a matrix with m rows and n columns filled in column-major order.
to_matrix_rowvector_int_int = lambda v, m, n: transpose(v.reshape(n, m))

# matrix to_matrix(matrix m, int m, int n, int col_major)
# Convert a matrix m to a matrix with m rows and n columns filled in row-major order if col_major equals 0 (otherwise, they get filled in column-major order).
def to_matrix_matrix_int_int_int(mat, m, n, col_major):
    if col_major == 0:
        mat.reshape(m, n)
    else:
        to_matrix_matrix_int_int(mat, m, n)


# matrix to_matrix(vector v, int m, int n, int col_major)
# Convert a vector v to a matrix with m rows and n columns filled in row-major order if col_major equals 0 (otherwise, they get filled in column-major order).
def to_matrix_vector_int_int_int(v, m, n, col_major):
    if col_major == 0:
        v.reshape(m, n)
    else:
        to_matrix_vector_int_int(v, m, n)


# matrix to_matrix(row_vector v, int m, int n, int col_major)
# Convert a row_vector a to a matrix with m rows and n columns filled in row-major order if col_major equals 0 (otherwise, they get filled in column-major order).
def to_matrix_rowvector_int_int_int(v, m, n, col_major):
    if col_major == 0:
        v.reshape(m, n)
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

## 8 Compound Arithmetic and Assignment

# Functions supported by the compiler

## 9 Higher-Order Functions

## 9.1 Algebraic equation solver

## 9.1.1 Specifying an algebraic equation as a function

# No function defined

## 9.1.2 Call to the algebraic solver

# vector algebra_solver(function algebra_system, vector y_guess, vector theta, real[] x_r, int[] x_i)
# Solves the algebraic system, given an initial guess, using the Powell hybrid algorithm.
algebra_solver_function_vector_vector_array_array = _XXX_TODO_XXX_("algebra_solver")

# vector algebra_solver(function algebra_system, vector y_guess, vector theta, real[] x_r, int[] x_i, real rel_tol, real f_tol, int max_steps)
# Solves the algebraic system, given an initial guess, using the Powell hybrid algorithm with additional control parameters for the solver.
algebra_solver_function_vector_vector_array_array_real_real_int = _XXX_TODO_XXX_("algebra_solver")
# XXX TODO: lifting to other types XXX

# vector algebra_solver_newton(function algebra_system, vector y_guess, vector theta, real[] x_r, int[] x_i)
# Solves the algebraic system, given an initial guess, using Newton’s method.
algebra_solver_newton_function_vector_vector_array_array = _XXX_TODO_XXX_("algebra_solver_newton")

# vector algebra_solver_newton(function algebra_system, vector y_guess, vector theta, real[] x_r, int[] x_i, real rel_tol, real f_tol, int max_steps)
# Solves the algebraic system, given an initial guess, using Newton’s method with additional control parameters for the solver.
algebra_solver_newton_function_vector_vector_array_array_real_real_int = _XXX_TODO_XXX_("algebra_solver_newton")
# XXX TODO: lifting to other types XXX

## 9.2 Ordinary Differential Equation (ODE) Solvers

## 9.2.1 Non-stiff solver

# vector[] ode_rk45(function ode, vector initial_state, real initial_time, real[] times, ...)
# Solves the ODE system for the times provided using the Dormand-Prince algorithm, a 4th/5th order Runge-Kutta method.
ode_rk45_function_vector_real_array = _XXX_TODO_XXX_("ode_rk45")
# XXX TODO: lifting to other types XXX

# vector[] ode_rk45_tol(function ode, vector initial_state, real initial_time, real[] times, real rel_tol, real abs_tol, int max_num_steps, ...)
# Solves the ODE system for the times provided using the Dormand-Prince algorithm, a 4th/5th order Runge-Kutta method with additional control parameters for the solver.
ode_rk45_tol_function_vector_real_array_real_real_int = _XXX_TODO_XXX_("ode_rk45_tol")
# XXX TODO: lifting to other types XXX

# vector[] ode_adams(function ode, vector initial_state, real initial_time, real[] times, ...)
# Solves the ODE system for the times provided using the Adams-Moulton method.
ode_adams_function_vector_real_array = _XXX_TODO_XXX_("ode_adams")
# XXX TODO: lifting to other types XXX

# vector[] ode_adams_tol(function ode, vector initial_state, real initial_time, real[] times, data real rel_tol, data real abs_tol, data int max_num_steps, ...)
# Solves the ODE system for the times provided using the Adams-Moulton method with additional control parameters for the solver.
ode_adams_tol_function_vector_real_array_real_real_int = _XXX_TODO_XXX_("ode_adams_tol")
# XXX TODO: lifting to other types XXX

## 9.2.2 Stiff solver

# vector[] ode_bdf(function ode, vector initial_state, real initial_time, real[] times, ...)
# Solves the ODE system for the times provided using the backward differentiation formula (BDF) method.
ode_bdf_function_real_array = _XXX_TODO_XXX_("ode_bdf")
# XXX TODO: lifting to other types XXX

# vector[] ode_bdf_tol(function ode, vector initial_state, real initial_time, real[] times, data real rel_tol, data real abs_tol, data int max_num_steps, ...)
# Solves the ODE system for the times provided using the backward differentiation formula (BDF) method with additional control parameters for the solver.
ode_bdf_tol_function_vector_real_array_real_real_int = _XXX_TODO_XXX_("ode_bdf_tol")
# XXX TODO: lifting to other types XXX

## 9.3 1D integrator

## 9.3.1 Specifying an integrand as a function

# No function defined

## 9.3.2 Call to the 1D integrator

# real integrate_1d (function integrand, real a, real b, real[] theta, real[] x_r, int[] x_i)
# Integrates the integrand from a to b.
integrate_1d_function = _XXX_TODO_XXX_("integrate_1d ")

# real integrate_1d (function integrand, real a, real b, real[] theta, real[] x_r, int[] x_i, real relative_tolerance)
# Integrates the integrand from a to b with the given relative tolerance.
integrate_1d_function = _XXX_TODO_XXX_("integrate_1d ")

## 9.4 Reduce-sum function

## 9.4.1 Specifying the reduce-sum function

# real reduce_sum(F f, T[] x, int grainsize, T1 s1, T2 s2, ...)
# real reduce_sum_static(F f, T[] x, int grainsize, T1 s1, T2 s2, ...)
# Returns the equivalent of f(x, 1, size(x), s1, s2, ...), but computes the result in parallel by breaking the array x into independent partial sums. s1, s2, ... are shared between all terms in the sum.
reduce_sum_function = _XXX_TODO_XXX_("reduce_sum")
reduce_sum_static_function = _XXX_TODO_XXX_("reduce_sum_static")

## 9.5 Map-rect function

## 9.5.1 Specifying the mapped function

# No function defined

## 9.5.2 Rectangular map

# vector map_rect(F f, vector phi, vector[] theta, data real[,] x_r, data int[,] x_i)
# Return the concatenation of the results of applying the function f, of type (vector, vector, real[], int[]):vector elementwise, i.e., f(phi, theta[n], x_r[n], x_i[n]) for each n in 1:N, where N is the size of the parallel arrays of job-specific/local parameters theta, real data x_r, and integer data x_r. The shared/global parameters phi are passed to each invocation of f.
map_rect_function = _XXX_TODO_XXX_("map_rect")

## 10 Deprecated Functions

## 10.1 integrate_ode_rk45, integrate_ode_adams, integrate_ode_bdf ODE integrators


## 10.1.1 Specifying an ordinary differential equation as a function

# No function defined

## 10.1.2 Non-stiff solver

# real[ , ] integrate_ode_rk45(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, real[] x_r, int[] x_i)
# Solves the ODE system for the times provided using the Dormand-Prince algorithm, a 4th/5th order Runge-Kutta method.
integrate_ode_rk45_function = _XXX_TODO_XXX_("integrate_ode_rk45")

# real[ , ] integrate_ode_rk45(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, real[] x_r, int[] x_i, real rel_tol, real abs_tol, int max_num_steps)
# Solves the ODE system for the times provided using the Dormand-Prince algorithm, a 4th/5th order Runge-Kutta method with additional control parameters for the solver.
integrate_ode_rk45_function = _XXX_TODO_XXX_("integrate_ode_rk45")

# real[ , ] integrate_ode(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, real[] x_r, int[] x_i)
# Solves the ODE system for the times provided using the Dormand-Prince algorithm, a 4th/5th order Runge-Kutta method.
integrate_ode_function = _XXX_TODO_XXX_("integrate_ode")

# real[ , ] integrate_ode_adams(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, data real[] x_r, data int[] x_i)
# Solves the ODE system for the times provided using the Adams-Moulton method.
integrate_ode_adams_function = _XXX_TODO_XXX_("integrate_ode_adams")

# real[ , ] integrate_ode_adams(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, data real[] x_r, data int[] x_i, data real rel_tol, data real abs_tol, data int max_num_steps)
# Solves the ODE system for the times provided using the Adams-Moulton method with additional control parameters for the solver.
integrate_ode_adams_function = _XXX_TODO_XXX_("integrate_ode_adams")

## 10.1.3 Stiff solver

# real[ , ] integrate_ode_bdf(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, data real[] x_r, data int[] x_i)
# Solves the ODE system for the times provided using the backward differentiation formula (BDF) method.
integrate_ode_bdf_function = _XXX_TODO_XXX_("integrate_ode_bdf")

# real[ , ] integrate_ode_bdf(function ode, real[] initial_state, real initial_time, real[] times, real[] theta, data real[] x_r, data int[] x_i, data real rel_tol, data real abs_tol, data int max_num_steps)
# Solves the ODE system for the times provided using the backward differentiation formula (BDF) method with additional control parameters for the solver.
integrate_ode_bdf_function = _XXX_TODO_XXX_("integrate_ode_bdf")
