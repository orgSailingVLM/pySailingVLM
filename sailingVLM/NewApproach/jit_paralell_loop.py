import time
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def prange_test(A):
    s = 0
    # Without "parallel=True" in the jit-decorator
    # the prange statement is equivalent to range
    for i in prange(A.shape[0]):
        s += A[i]
    return s

start = time.time()
N = 100000000
a = np.arange(0, N)

#print(a.shape[0])
prange_test(a)
end = time.time()

    
print("Elapsed (with compilation) = %s" % (end - start))
