import numba
import numpy as np
#from numpy.linalg import norm
from typing import Tuple


@numba.jit(nopython=True, cache=True, debug=True)
def is_in_vortex_core(vector_list : numba.typed.List) -> bool:
    """
    is_in_vortex_core check if list of vectors is n vortex core

    :param numba.typed.List vector_list: list of vectors
    :return bool: True or False
    """

    for vec in vector_list:
        if np.linalg.norm(vec) < 1e-8:
            return True
    return False


@numba.jit(numba.float64[::1](numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.optional(numba.float64)), nopython=True, debug = True, cache=True) 
def vortex_line(p: np.array, p1: np.array, p2: np.array, gamma: float = 1.0) -> np.array:
    """
    vortex_line vortex line by Katz & Plotkin p 254

    :param np.array p: point at which  calculation is done
    :param np.array p1: point 1
    :param np.array p2: point 2
    :param float gamma: gamma orientation, defaults to 1.0
    :return np.array: velocity component
    """
    # strona 254

    r0 = np.asarray(p2 - p1)
    r1 = np.asarray(p - p1)
    r2 = np.asarray(p - p2)
    
    r1_cross_r2 = np.cross(r1, r2)
    
    q_ind = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # in nonpython mode must be list reflection to convert list to non python type
    # nested python oject can be badly converted -> recommend to use numba.typed.List
    # if r1 or r2 or |r1_cross_r2|^2 < epsilon
    # convert float - |r1_cross_r2|^2 to array with 1 element
    # this is due to numba
    # numba do not understand typed list with 2 vectors (r1 nad r2) and scalar like float
    sq = np.array([np.square(np.linalg.norm(r1_cross_r2))])
    b = is_in_vortex_core(numba.typed.List([r1, r2, sq]))

    if b:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q_ind = r1_cross_r2 / np.square(np.linalg.norm(r1_cross_r2))
        q_ind *= np.dot(r0, (r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2)))
        q_ind *= gamma / (4 * np.pi)

    return q_ind


@numba.jit(numba.float64[::1](numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.optional(numba.float64)), nopython=True, debug = True, cache=True) 
# ten sposob jest wolniejszy niz wskazanie wprost typow
#@numba.jit(nopython=True, cache=True)
def vortex_infinite_line(P: np.ndarray, A: np.array, r0: np.ndarray, gamma : float = 1.0) -> np.ndarray:
    """
    vortex_infinite_line vortex infinite line

    :param np.ndarray P: point P
    :param np.array A: point A
    :param np.ndarray r0: r0 vector
    :param float gamma: gamma orientation, defaults to 1.0
    :return np.ndarray: velocity component
    """

    u_inf = r0 / np.linalg.norm(r0)
    ap = P - A
    norm_ap = np.linalg.norm(ap)

    v_ind = np.cross(u_inf, ap) / (
                norm_ap * (norm_ap - np.dot(u_inf, ap)))  # todo: consider checking is_in_vortex_core
    v_ind *= gamma / (4. * np.pi)
    return v_ind


@numba.jit(numba.float64[::1](numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.optional(numba.float64)), nopython=True, debug = True, cache=True) 
def vortex_horseshoe(p: np.array, B: np.array, C: np.array, V_app_infw: np.ndarray,
                        gamma: float = 1.0) -> np.array:
    """
    vortex_horseshoe _summary_

    :param np.array p: point form which calculation is done
    :param np.array B: point B
    :param np.array C: point C
    :param np.ndarray V_app_infw: apparent wind velocity for infinite sail
    :param float gamma: gamma orientation, defaults to 1.0
    :return np.array: velocity component
    
    
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

# @numba.jit(nopython=True, cache=True) daje ten sam wynik co:
@numba.jit(numba.float64[::1](numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.optional(numba.float64)), nopython=True, debug = True, cache=True) 
def vortex_horseshoe_v2(p: np.array, A: np.array, B: np.array, C: np.array, D: np.array, V_app_infw : np.ndarray,
                gamma: float = 1.0) -> np.array:
    """
    vortex_ring vortex ring

    :param np.array p: point form which calculation is done
    :param np.array A: point A
    :param np.array B: point B
    :param np.array C: point C
    :param np.array D: point D
    :param np.ndarray V_app_infw: apparent wind velocity for infinite sail
    :param float gamma: gamma orientation, defaults to 1.0
    :return np.array: velocity component
    """
    sub1 = vortex_line(p, A, B, gamma)
    sub2 = vortex_line(p, B, C, gamma)
    sub3 = vortex_line(p, C, D, gamma)
    #sub4 = vortex_line(p, D, A, gamma)
    # wczesniej od C teraz od D
    inf_line_1 = vortex_infinite_line(p, D, V_app_infw, gamma)
    # wczesniej od B teraz od A
    inf_line_2 = vortex_infinite_line(p, A, V_app_infw, -1.0 * gamma)
    
    q_ind = sub1 + sub2 + sub3 + inf_line_1 + inf_line_2
    return q_ind


# @numba.jit(nopython=True, cache=True) daje ten sam wynik co:
@numba.jit(numba.float64[::1](numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.optional(numba.float64)), nopython=True, debug = True, cache=True) 
def vortex_ring(p: np.array, A: np.array, B: np.array, C: np.array, D: np.array,
                gamma: float = 1.0) -> np.array:
    """
    vortex_ring vortex ring

    :param np.array p: point form which calculation is done
    :param np.array A: point A
    :param np.array B: point B
    :param np.array C: point C
    :param np.array D: point D
    :param float gamma: gamma orientation, defaults to 1.0
    :return np.array: velocity component
    """
    sub1 = vortex_line(p, A, B, gamma)
    sub2 = vortex_line(p, B, C, gamma)
    sub3 = vortex_line(p, C, D, gamma)
    sub4 = vortex_line(p, D, A, gamma)

    q_ind = sub1 + sub2 + sub3 + sub4
    return q_ind


# parallel - no time change
numba.jit(numba.float64[:, ::1](numba.float64[:, :, ::1], numba.float64[::1]), nopython=True, debug = True, cache=True) 
#@numba.jit(numba.float64[:, ::1](numba.float64[:, ::1], numba.float64[::1]), nopython=True, debug = True, cache=True) 
#@numba.jit(nopython=True, cache=True)
def calc_induced_velocity(v_ind_coeff : np.ndarray, gamma_magnitude : float) -> np.ndarray:
    """
    calc_induced_velocity calculate induced velocity

    :param np.ndarray v_ind_coeff: velocity induced ceofs
    :param float gamma_magnitude: gamma magnitude
    :return np.ndarray: induced wind velocity
    """
    N = gamma_magnitude.shape[0]

    V_induced = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            V_induced[i] += v_ind_coeff[i,j] * gamma_magnitude[j]

    return V_induced


def calculate_app_fs(V_app_infs : np.ndarray, v_ind_coeff : np.ndarray, gamma_magnitude : float) -> Tuple[np.ndarray, np.ndarray]:
    """
    calculate_app_fs calculate apparent wind velocity for finite sail

    :param np.ndarray V_app_infs: apparent wind velocity for infinite sail
    :param np.ndarray v_ind_coeff: velocity induced coefs
    :param float gamma_magnitude: gamma magnitude
    :return Tuple[np.ndarray, np.ndarray]: wind velocity induced, wind velocity apparent finite sail
    """
    V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
    V_app_fs = V_app_infs + V_induced
    return V_induced, V_app_fs



