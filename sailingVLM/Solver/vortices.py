
import numpy as np
from numpy.linalg import norm
import numba

@numba.jit(numba.float64[::1](numba.float64[::1]), nopython=True, debug=False)
def normalize(x):
    # xn = x / norm(x)
    xn = x / np.linalg.norm(x)
    return xn
