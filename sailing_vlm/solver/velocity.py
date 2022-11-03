import numba
import numpy as np
from numpy.linalg import norm


#@numba.jit(nopython=True, cache=True)
def is_in_vortex_core(vector_list : numba.typed.List) -> bool:
    """
    is_in_vortex_core check if list of vectors is n vortex core

    :param numba.typed.List vector_list: list of vectors
    :return bool: True or False
    """
    #todo: polepszyc to
    for vec in vector_list:
        if norm(vec) < 1e-9:
            return True
    return False


#@numba.jit(nopython=True) #-> slower than version below
#@numba.jit(numba.float64[::1](numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.optional(numba.float64)), nopython=True, debug = True, cache=True) 
def vortex_line(p: np.array, p1: np.array, p2: np.array, gamma: float = 1.0) -> np.array:
#def vortex_line(p, p1,  p2,  gamma = 1.0):
    # strona 254

    r0 = np.asarray(p2 - p1)
    r1 = np.asarray(p - p1)
    r2 = np.asarray(p - p2)
    
    r1_cross_r2 = np.cross(r1, r2)
    
    q_ind = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # in nonpython mode must be list reflection to convert list to non python type
    # nested python oject can be badly converted -> recommend to use numba.typed.List
    b = is_in_vortex_core(numba.typed.List([r1, r2, r1_cross_r2]))
    
    # for normal code without numba
    #b = is_in_vortex_core([r1, r2, r1_cross_r2])
    if b:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q_ind = r1_cross_r2 / np.square(np.linalg.norm(r1_cross_r2))
        q_ind *= np.dot(r0, (r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2)))
        q_ind *= gamma / (4 * np.pi)

    return q_ind

# 6 seconds less with it (tested on 30x30)
#@numba.jit(nopython=True, cache=True)
def vortex_infinite_line(P: np.ndarray, A: np.array, r0: np.ndarray, gamma : float = 1.0):

    u_inf = r0 / norm(r0)
    ap = P - A
    norm_ap = norm(ap)

    v_ind = np.cross(u_inf, ap) / (
                norm_ap * (norm_ap - np.dot(u_inf, ap)))  # todo: consider checking is_in_vortex_core
    v_ind *= gamma / (4. * np.pi)
    return v_ind

# numba works slower here (40x40)
#@numba.jit(nopython=True)
def vortex_horseshoe(p: np.array, B: np.array, C: np.array, V_app_infw: np.ndarray,
                        gamma: float = 1.0) -> np.array:
    """
    B ------------------ +oo
    |
    |
    C ------------------ +oo
    """
    sub1 = vortex_infinite_line(p, C, V_app_infw, gamma)
    sub2 = vortex_line(p, B, C, gamma)
    sub3 = vortex_infinite_line(p, B, V_app_infw, -1.0 * gamma)
    q_ind = sub1 + sub2 + sub3
    return q_ind

#@numba.jit(nopython=True, cache=True)
def vortex_ring(p: np.array, A: np.array, B: np.array, C: np.array, D: np.array,
                gamma: float = 1.0) -> np.array:

    sub1 = vortex_line(p, A, B, gamma)
    #assert not vortex_line.nopython_signatures
    sub2 = vortex_line(p, B, C, gamma)
    sub3 = vortex_line(p, C, D, gamma)
    sub4 = vortex_line(p, D, A, gamma)

    q_ind = sub1 + sub2 + sub3 + sub4
    return q_ind


# parallel - no time change
#@numba.jit(nopython=True, cache=True)
def calc_induced_velocity(v_ind_coeff, gamma_magnitude):
    N = gamma_magnitude.shape[0]
    
    t2 = len(gamma_magnitude)
    V_induced = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            V_induced[i] += v_ind_coeff[i,j] * gamma_magnitude[j]

    return V_induced


def calculate_app_fs(V_app_infs, v_ind_coeff, gamma_magnitude):
    V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
    V_app_fs = V_app_infs + V_induced
    return V_induced, V_app_fs

