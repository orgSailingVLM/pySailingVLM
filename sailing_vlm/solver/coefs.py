import numpy as np
import pandas as pd
from typing import List, Tuple
from numpy.linalg import norm

from sailing_vlm.solver import velocity

import numba


def get_leading_edge_mid_point(p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    return (p2 + p3) / 2.

def get_trailing_edge_mid_points(p1: np.ndarray, p4: np.ndarray) -> np.ndarray:
    return (p4 + p1) / 2.

def calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels: np.ndarray, gamma_orientation : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points _summary_

    :param np.ndarray panels: panels
    :param float gamma_orientation: gamma orientation
    :return Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: normals, collocation points, center of pressure, rings, span vectors, leading mid points, trailing edge mid points
    """
    K = panels.shape[0]
    ns = np.zeros((K, 3))
    span_vectors = np.zeros((K, 3))
    collocation_points = np.zeros((K, 3))
    center_of_pressure = np.zeros((K, 3))
    leading_mid_points = np.zeros((K, 3))
    trailing_edge_mid_points = np.zeros((K, 3))
    
    rings = np.zeros(shape=panels.shape)
    for idx, panel in enumerate(panels):
        p1 = panel[0]
        p2 = panel[1]
        p3 = panel[2]
        p4 = panel[3]

        vect_A = p4 - p2
        vect_B = p3 - p1

        leading_mid_points[idx] = get_leading_edge_mid_point(p2, p3)
        trailing_edge_mid_points[idx] = get_trailing_edge_mid_points(p1, p4)
        dist = trailing_edge_mid_points[idx] - leading_mid_points[idx]

        collocation_points[idx] = leading_mid_points[idx] + 0.75 * dist
        center_of_pressure[idx] = leading_mid_points[idx] + 0.25 * dist

        p2_p1 = p1 - p2
        p3_p4 = p4 - p3

        A = p1 + p2_p1 / 4.
        B = p2 + p2_p1 / 4.
        C = p3 + p3_p4 / 4.
        D = p4 + p3_p4 / 4.

        rings[idx] = np.array([A, B, C, D])
        
        # span vectors
        bc = C - B
        bc *= gamma_orientation
        span_vectors[idx] = bc
        n = np.cross(vect_A, vect_B)
        n = n / np.linalg.norm(n)
        ns[idx] = n
    return ns, collocation_points, center_of_pressure, rings, span_vectors, leading_mid_points, trailing_edge_mid_points


# sails = [jib, main]

# tak bylo w starej funkcji:
# # numba tutaj nie rozumie typow -> do poprawki
# #@numba.jit(nopython=True)
# #@numba.njit(parallel=True)
def get_influence_coefficients_spanwise(collocation_points: np.ndarray, rings: np.ndarray, normals: np.ndarray, V_app_infw: np.ndarray, horseshoe_info : np.ndarray, gamma_orientation : float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    get_influence_coefficients_spanwise calculate coefs spanwise

    :param np.ndarray collocation_points: collocation points
    :param np.ndarray rings: rings
    :param np.ndarray normals: normals
    :param np.ndarray V_app_infw: wind apparent infinite sail velocity
    :param np.ndarray horseshoe_info: if panel is hourseshoe, ten horseshoe_info is True 
    :param float gamma_orientation: gamma orientation, defaults to 1.0
    :return Tuple[np.ndarray, np.ndarray, np.ndarray]: coefs, RHS, wind coefs
    """
    
    m = collocation_points.shape[0]
    # to nie dziala dla jiba?
    RHS = -V_app_infw.dot(normals.transpose()).diagonal()
    #RHS = [-np.dot(V_app_infw[i], normals[i]) for i in range(normals.shape[0])]
    coefs = np.zeros((m, m))
    wind_coefs = np.zeros((m, m, 3))
    
    # loop over other vortices
    for i, ring in enumerate(rings):
        A = ring[0]
        B = ring[1]
        C = ring[2]
        D = ring[3]
        # loop over points
        for j, point in enumerate(collocation_points):
           
            a = velocity.vortex_ring(point, A, B, C, D, gamma_orientation)
            # poprawka na trailing edge
            # todo: zrobic to w drugim, oddzielnym ifie
            # poziomo od 0 do n-1, reszta odzielnie
            if horseshoe_info[i]:
                #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                a = velocity.vortex_horseshoe(point, ring[1], ring[2], V_app_infw[i], gamma_orientation)
            b = np.dot(a, normals[j].reshape(3, 1))
            wind_coefs[j, i] = a
            coefs[j, i] = b
    RHS = np.asarray(RHS)
                
    return coefs, RHS, wind_coefs

def solve_eq(coefs: np.ndarray, RHS: np.ndarray) -> np.ndarray:
    """
    solve_eq solve equation for gamma 

    :param np.ndarray coefs: coefs
    :param np.ndarray RHS: RHS
    :return np.ndarray: calculated gamma
    """
    big_gamma = np.linalg.solve(coefs, RHS)
    return big_gamma



# do przykladu z aircraftem
def get_vlm_CL_CD_free_wing(F: np.ndarray, V: np.array, rho : float, S : float) -> Tuple[float, float]:
    
    total_F = np.sum(F, axis=0)
    q = 0.5 * rho * (np.linalg.norm(V) ** 2) * S
    CL_vlm = total_F[2] / q
    CD_vlm = total_F[0] / q
    
    return CL_vlm, CD_vlm


def get_CL_CD_free_wing(AR, AoA_deg):
    #  TODO allow tapered wings in book coeff_formulas
    a0 = 2. * np.pi  # dCL/d_alfa in 2D [1/rad]
    e_w = 0.8  # span efficiency factor, range: 0.8 - 1.0

    a = a0 / (1. + a0 / (np.pi * AR * e_w))

    CL_expected_3d = a * np.deg2rad(AoA_deg)
    CD_ind_expected_3d = CL_expected_3d * CL_expected_3d / (np.pi * AR * e_w)

    return CL_expected_3d, CD_ind_expected_3d