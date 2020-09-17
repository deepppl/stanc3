from torch import sqrt, exp, log, log10, square, ones

def rep_vector(x, m):
    return x * ones(m)

def rep_row_vector(x, m):
    return x * ones(1, m)

def rep_matrix(x, *dims):
    if len(dims) == 2:
        return x * ones(*dims)
    elif len(x.shape) == 1:
        return x.expand(x.shape[0], *dims)
    elif len(x.shape) == 2:
        return x.expand(*dims, x.shape[1])
    else:
        assert False: 'rep_matrix: bad dimensions'
