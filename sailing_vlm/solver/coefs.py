import numpy as np
import os
import pandas as pd
from typing import List, Tuple
from numpy.linalg import norm

import numba

from sailing_vlm.solver.velocity import vortex_ring, vortex_horseshoe, vortex_horseshoe_infinite_components, vortex_ring_components

# parallel slows down beacuse loop is not big
@numba.jit(numba.types.Tuple((numba.float64[:, ::1], numba.float64[:, :, ::1])) (numba.float64[:, ::1], numba.float64[:, ::1], numba.float64[:, :, ::1], numba.float64[:, ::1], numba.boolean[::1], numba.float64), nopython=True, debug = True, cache=True)
# @numba.jit(numba.types.Tuple((numba.float64[:, ::1], numba.float64[:, ::1])) (numba.float64[:, ::1], numba.float64[:, ::1], numba.float64[:, :, ::1], numba.float64[:, ::1], numba.boolean[::1], numba.float64), nopython=True, debug = True, cache=True)
def calc_wind_coefs(V_app_infw, points_for_calculations, rings, normals, trailing_edge_info : np.ndarray, gamma_orientation : np.ndarray):

    m = points_for_calculations.shape[0]

    coefs = np.zeros((m,m))

    #bs = np.zeros((m, m))
    wind_coefs = np.zeros((m, m, 3))
    for i in range(points_for_calculations.shape[0]):

        # loop over other vortices
        for j in range(rings.shape[0]):
            A = rings[j][0]
            B = rings[j][1]
            C = rings[j][2]
            D = rings[j][3]
            
            a = vortex_ring(points_for_calculations[i], A, B, C, D, gamma_orientation)
            #b = vortex_ring_components(points_for_calculations[i], A, B, C, D, gamma_orientation)
            # poprawka na trailing edge
            # todo: zrobic to w drugim, oddzielnym ifie
            if trailing_edge_info[j]:
                a = vortex_horseshoe(points_for_calculations[i], B, C, V_app_infw[j], gamma_orientation)
                #b = vortex_horseshoe_infinite_components(points_for_calculations[i], B, C, V_app_infw[j], gamma_orientation) 
            coefs[i, j] = np.dot(a, normals[i].reshape(3, 1))[0]
            # this is faster than wind_coefs[i, j, :] = a around 0.1s (case 10x10)
            
            
            wind_coefs[i, j, 0] = a[0]
            wind_coefs[i, j, 1] = a[1]
            wind_coefs[i, j, 2] = a[2]
            
            #bs[i, j] = np.dot(b, normals[i].reshape(3, 1))[0]
            #wind_coefs[i, j, 1] = bb[1]
            #wind_coefs[i, j, 2] = bb[2]
    
            
    return coefs, wind_coefs


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

        # tutaj bylo zmienione!!!
        #vect_A = p4 - p2
        #vect_B = p3 - p1
        
        vect_B = p4 - p2
        vect_A = p3 - p1

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

def calculate_stuff(panels: np.ndarray, leading_edge_info : np.ndarray, gamma_orientation : float, n_chordwise : int, n_spanwise: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points _summary_

    :param np.ndarray panels: panels
    :param float gamma_orientation: gamma orientation
    :param int n_chordwise: number of panels chordwise
    :param int n_spanwise: number of panels spanwise
    :return Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: normals, collocation points, center of pressure, rings, span vectors, leading mid points, trailing edge mid points
    """
    K = panels.shape[0]
    # ns = np.zeros((K, 3))
    # span_vectors = np.zeros((K, 3))
    # collocation_points = np.zeros((K, 3))
    # center_of_pressure = np.zeros((K, 3))
    # leading_mid_points = np.zeros((K, 3))
    # trailing_edge_mid_points = np.zeros((K, 3))
    
    
    
    # jib, main, jib underwater, main underwater - 4 times panels grid
    # but if someone like only jib?, then calculate number of grids like this
    # G = int (K / (n_chordwise * n_spanwise))
    # panels_splitted = np.array_split(panels, G)
    # rings = np.zeros(shape=(G, n_chordwise, n_spanwise, 4, 3))

    # ns = np.zeros(shape=(G, n_chordwise, n_spanwise, 3))
    # span_vectors = np.zeros((G, n_chordwise, n_spanwise, 3))
    # collocation_points = np.zeros((G, n_chordwise, n_spanwise, 3))
    # center_of_pressure = np.zeros((G, n_chordwise, n_spanwise, 3))
    # leading_mid_points = np.zeros((G, n_chordwise, n_spanwise, 3))
    # trailing_edge_mid_points = np.zeros((G, n_chordwise, n_spanwise, 3))
    
    G = int (K / (n_chordwise * n_spanwise))
    panels_splitted = np.array_split(panels, G)
    leading_edge_info_splitted = np.array_split(leading_edge_info, G)
    
    rings = np.zeros(shape=(G, n_chordwise * n_spanwise, 4, 3))
    ns = np.zeros(shape=(G, n_chordwise * n_spanwise, 3))
    span_vectors = np.zeros((G, n_chordwise * n_spanwise, 3))
    collocation_points = np.zeros((G, n_chordwise * n_spanwise, 3))
    center_of_pressure = np.zeros((G, n_chordwise * n_spanwise, 3))
    leading_mid_points = np.zeros((G, n_chordwise * n_spanwise, 3))
    trailing_edge_mid_points = np.zeros((G, n_chordwise * n_spanwise, 3))
    
    for idx, (grid_panels, leading_grid_info) in enumerate(zip(panels_splitted,leading_edge_info_splitted)):
       # grid_panels = grid_panels.reshape(n_chordwise, n_spanwise, 4, 3)
        #grid_panels = grid_panels.reshape(n_spanwise, n_chordwise, 4, 3)
        n_panels = grid_panels.shape[0]
        for k in range(n_panels):
        #for i in range(n_chordwise):
        #    for j in range(n_spanwise):
                # current panel
            panel = grid_panels[k]
            p1 = panel[0]
            p2 = panel[1]
            p3 = panel[2]
            p4 = panel[3]
            
            vect_B = p4 - p2
            vect_A = p3 - p1

            leading_mid_points[idx, k] = get_leading_edge_mid_point(p2, p3)
            trailing_edge_mid_points[idx, k] = get_trailing_edge_mid_points(p1, p4)
            dist = trailing_edge_mid_points[idx, k] - leading_mid_points[idx, k]

            collocation_points[idx, k] = leading_mid_points[idx, k] + 0.75 * dist
            center_of_pressure[idx, k] = leading_mid_points[idx, k] + 0.25 * dist

            p2_p1 = p1 - p2
            p3_p4 = p4 - p3
            
            n = np.cross(vect_A, vect_B)
            n = n / np.linalg.norm(n)
            ns[idx, k] = n
            
            if leading_grid_info[k]:  
            #if i == 0:
                # leading edge
                A = p1 + p2_p1 / 4.
                B = p2 + p2_p1 / 4.
                C = p3 + p3_p4 / 4.
                D = p4 + p3_p4 / 4.
            else:
                # not leading edge
                A = p1 + p2_p1 / 4.
                B = rings[idx, k - n_spanwise][0]
                C = rings[idx, k -n_spanwise][3]
                D = p4 + p3_p4 / 4.
                
                
            rings[idx, k] = np.array([A, B, C, D])
            # span vectors
            bc = C - B
            bc *= gamma_orientation
            span_vectors[idx,k] = bc

    #K2 = G*n_chordwise*n_spanwise
    span_vectors = span_vectors.reshape(K,3)
    rings = rings.reshape(K, 4, 3)

    ns =ns.reshape(K,3)

    collocation_points = collocation_points.reshape(K,3)
    center_of_pressure = center_of_pressure.reshape(K,3)
    leading_mid_points = leading_mid_points.reshape(K,3)
    trailing_edge_mid_points = trailing_edge_mid_points.reshape(K,3)         
    
    return ns, collocation_points, center_of_pressure, rings, span_vectors, leading_mid_points, trailing_edge_mid_points

# sails = [jib, main]

def calculate_RHS(V_app_infw, normals):
    RHS = -V_app_infw.dot(normals.transpose()).diagonal()
    RHS = np.asarray(RHS)
    return RHS

# tak bylo w starej funkcji:
# # numba tutaj nie rozumie typow -> do poprawki
# #@numba.jit(nopython=True)
# #@numba.njit(parallel=True)
def get_influence_coefficients_spanwise(collocation_points: np.ndarray, rings: np.ndarray, normals: np.ndarray, V_app_infw: np.ndarray, trailing_edge_info : np.ndarray, gamma_orientation : float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    get_influence_coefficients_spanwise calculate coefs spanwise

    :param np.ndarray collocation_points: collocation points
    :param np.ndarray rings: rings
    :param np.ndarray normals: normals
    :param np.ndarray V_app_infw: wind apparent infinite sail velocity
    :param np.ndarray trailing_edge_info: if panel is hourseshoe, ten horseshoe_info is True 
    :param float gamma_orientation: gamma orientation, defaults to 1.0
    :return Tuple[np.ndarray, np.ndarray, np.ndarray]: coefs, RHS, wind coefs
    """

    coefs, wind_coefs = calc_wind_coefs(V_app_infw, collocation_points, rings, normals, trailing_edge_info, gamma_orientation)
    RHS = calculate_RHS(V_app_infw, normals)

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

