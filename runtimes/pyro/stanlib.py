from torch import sqrt, exp, log, log10, square, ones

def rep_vector(x, m):
    return x * ones(m)

def rep_row_vector(x, m):
    return x * ones(1,m)
