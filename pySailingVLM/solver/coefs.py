from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pySailingVLM.solver.vlm import Vlm
    from pySailingVLM.runner.container import Solver
    
import numpy as np
import os
import pandas as pd
from typing import List, Tuple
from numpy.linalg import norm

import numba

from pySailingVLM.solver.velocity import vortex_ring, vortex_horseshoe, vortex_horseshoe_v2

# parallel slows down beacuse loop is not big
@numba.jit(numba.types.Tuple((numba.float64[:, ::1], numba.float64[:, :, ::1])) (numba.float64[:, ::1], numba.float64[:, ::1], numba.float64[:, :, ::1], numba.float64[:, ::1], numba.boolean[::1], numba.float64), nopython=True, debug = True, cache=True)
# @numba.jit(numba.types.Tuple((numba.float64[:, ::1], numba.float64[:, ::1])) (numba.float64[:, ::1], numba.float64[:, ::1], numba.float64[:, :, ::1], numba.float64[:, ::1], numba.boolean[::1], numba.float64), nopython=True, debug = True, cache=True)
def calc_velocity_coefs(V_app_infw, points_for_calculations, rings, normals, trailing_edge_info : np.ndarray, gamma_orientation : np.ndarray):
    m = points_for_calculations.shape[0]

    coefs = np.zeros((m,m)) # coefs calculated for normalized velocity
    v_ind_coeff = np.zeros((m, m, 3))
    for i in range(points_for_calculations.shape[0]):

        # loop over other vortices
        for j in range(rings.shape[0]):
            A = rings[j][0]
            B = rings[j][1]
            C = rings[j][2]
            D = rings[j][3]
            
            a = vortex_ring(points_for_calculations[i], A, B, C, D, gamma_orientation)
            # poprawka na trailing edge
            # todo: zrobic to w drugim, oddzielnym ifie
            if trailing_edge_info[j]:
                #a = vortex_horseshoe(points_for_calculations[i], B, C, V_app_infw[j], gamma_orientation)
                a = vortex_horseshoe_v2(points_for_calculations[i], A, B, C, D,V_app_infw[j], gamma_orientation)
            coefs[i, j] = np.dot(a, normals[i].reshape(3, 1))[0]
            # this is faster than wind_coefs[i, j, :] = a around 0.1s (case 10x10)
            
            v_ind_coeff[i, j, 0] = a[0] # FOR NUMBA
            v_ind_coeff[i, j, 1] = a[1]
            v_ind_coeff[i, j, 2] = a[2]
            
    return coefs, v_ind_coeff

# do wywalenia pozniej
def get_leading_edge_mid_point(p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    return (p2 + p3) / 2.


# do wywalenia pozniej
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
    ctr_p = np.zeros((K, 3))
    cp = np.zeros((K, 3))
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

        ctr_p[idx] = leading_mid_points[idx] + 0.75 * dist
        cp[idx] = leading_mid_points[idx] + 0.25 * dist

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
    return ns, ctr_p, cp, rings, span_vectors, leading_mid_points, trailing_edge_mid_points

def calculate_vlm_variables(panels: np.ndarray, trailing_edge_info : np.ndarray, gamma_orientation : float, n_chordwise : int, n_spanwise: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    calculaye_vlm_variables calculate normals, control points, center of pressure, rings, span vectors, leading mid points, trailing edge mid points

    :param np.ndarray panels: panels
    :param float gamma_orientation: gamma orientation
    :param int n_chordwise: number of panels chordwise
    :param int n_spanwise: number of panels spanwise
    :return Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: normals, collocation points, center of pressure, rings, span vectors, leading mid points, trailing edge mid points
    """
    K = panels.shape[0]
    G = int (K / (n_chordwise * n_spanwise))
    panels_splitted = np.array_split(panels, G)
    trailing_edge_info_splitted = np.array_split(trailing_edge_info, G)
    
    rings = np.zeros(shape=(G, n_chordwise * n_spanwise, 4, 3))
    ns = np.zeros(shape=(G, n_chordwise * n_spanwise, 3))
    span_vectors = np.zeros((G, n_chordwise * n_spanwise, 3))
    ctr_p = np.zeros((G, n_chordwise * n_spanwise, 3))
    cp = np.zeros((G, n_chordwise * n_spanwise, 3))
    leading_mid_points = np.zeros((G, n_chordwise * n_spanwise, 3))
    trailing_edge_mid_points = np.zeros((G, n_chordwise * n_spanwise, 3))
    
    for idx, (grid_panels, trailing_grid_info) in enumerate(zip(panels_splitted,trailing_edge_info_splitted)):

        n_panels = grid_panels.shape[0]
        for k in range(n_panels):

            currnet_panel = grid_panels[k]
            c_p1 = currnet_panel[0]
            c_p2 = currnet_panel[1]
            c_p3 = currnet_panel[2]
            c_p4 = currnet_panel[3]
            
            vect_B = c_p4 - c_p2
            vect_A = c_p3 - c_p1

            leading_mid_points[idx, k] = (c_p2 + c_p3) / 2.0 
            trailing_edge_mid_points[idx, k] = (c_p1 + c_p4) / 2.0 
            dist = trailing_edge_mid_points[idx, k] - leading_mid_points[idx, k]

            ctr_p[idx, k] = leading_mid_points[idx, k] + 0.75 * dist
            cp[idx, k] = leading_mid_points[idx, k] + 0.25 * dist

            c_p2_p1 = c_p1 - c_p2
            c_p3_p4 = c_p4 - c_p3
            
            B = c_p2 + c_p2_p1 / 4.
            C = c_p3 + c_p3_p4 / 4.
                
            n = np.cross(vect_A, vect_B)
            n = n / np.linalg.norm(n)
            ns[idx, k] = n
            
            if trailing_grid_info[k]:  
                # if trailing edge
                A = c_p1 + c_p2_p1 / 4.
                D = c_p4 + c_p3_p4 / 4.
            else:
                # if not trailing edge
                next_panel = grid_panels[k + n_spanwise]
                n_p1 = next_panel[0]
                n_p2 = next_panel[1]
                n_p3 = next_panel[2]
                n_p4 = next_panel[3]

                n_p2_p1 = n_p1 - n_p2
                n_p3_p4 = n_p4 - n_p3
                
                n_B = n_p2 + n_p2_p1 / 4.
                n_C = n_p3 + n_p3_p4 / 4.
            
                A = n_B
                D = n_C
                
            rings[idx, k] = np.array([A, B, C, D])
            # span vectors
            bc = C - B
            bc *= gamma_orientation
            span_vectors[idx,k] = bc


    span_vectors = span_vectors.reshape(K,3)
    rings = rings.reshape(K, 4, 3)

    ns =ns.reshape(K,3)

    ctr_p = ctr_p.reshape(K,3)
    cp = cp.reshape(K,3)
    leading_mid_points = leading_mid_points.reshape(K,3)
    trailing_edge_mid_points = trailing_edge_mid_points.reshape(K,3)         
    
    return ns, ctr_p, cp, rings, span_vectors, leading_mid_points, trailing_edge_mid_points


def calculate_RHS(V_app_infw, normals):
    RHS = -V_app_infw.dot(normals.transpose()).diagonal()
    RHS = np.asarray(RHS)
    return RHS


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

def get_vlm_Cxyz(F: np.ndarray, V: np.array, rho : float, S : float) -> Tuple[float, float, float]:
    
    total_F = np.sum(F, axis=0)
    q = 0.5 * rho * (np.linalg.norm(V) ** 2) * S
    #Cx_vlm, Cy_vlm, Cz_vlm = total_F / q
    
    return total_F / q #Cx_vlm, Cy_vlm, Cz_vlm

def get_CL_CD_free_wing(AR, AoA_deg):
    #  TODO allow tapered wings in book coeff_formulas
    a0 = 2. * np.pi  # dCL/d_alfa in 2D [1/rad]
    e_w = 0.8  # span efficiency factor, range: 0.8 - 1.0

    a = a0 / (1. + a0 / (np.pi * AR * e_w))

    CL_expected_3d = a * np.deg2rad(AoA_deg)
    CD_ind_expected_3d = CL_expected_3d * CL_expected_3d / (np.pi * AR * e_w)

    return CL_expected_3d, CD_ind_expected_3d


def get_C(panels: np.ndarray, areas: np.ndarray, force : np.ndarray, wind : np.ndarray, n_spanwise: int, n_chordwise: int, rho: float):
    # k number of sails, 1 (jb or main), 2 (jib and main)
    k = int(panels.shape[0] / (n_spanwise * n_chordwise * 2) ) # *2 in denominator due to underwater part
    
    # sail_forces - list of length = number of sails above water
    # each item in list is an array with forces for all panels in sail
    # winds, areas - same
    sail_winds = np.split(wind, (2*k))
    sail_forces = np.split(force, (2*k))
    sail_areas = np.split(areas, (2*k))
    sails_C = np.zeros((k, n_spanwise), dtype=float)
    
    # iterate through above water sails
    for i in range(k):
        # coefficient for all panels for specific sail
        c = np.linalg.norm(sail_forces[i], axis=1) / (0.5 * rho * np.linalg.norm(sail_winds[i], axis=1) ** 2 * sail_areas[i])
    
        section_c_list = np.zeros((n_spanwise,), dtype=float)
        
        # iterate through strips
        for j in range(n_spanwise):
            # coefficients for strip
            section_cs = c.reshape(n_chordwise, n_spanwise)[:,j]
            # areas for strip
            section_sail_areas = sail_areas[i].reshape(n_chordwise, n_spanwise)[:,j]
            
            # total section coefficient per strip
            section_c = np.sum(section_cs * section_sail_areas) / np.sum(section_sail_areas)
            section_c_list[j]= section_c

        sails_C[i] = section_c_list
    return sails_C


def get_data_for_coeff_plot(myvlm : Vlm, solver : Solver) -> Tuple[np.ndarray, list, list]:
    """
    get_data_for_coeff_plot get x and y data for section lift and drag coefficient

    :param Vlm myvlm: Vlm object
    :param Solver solver: Solver object
    :return Tuple[np.ndarray, list, list]: mean cp.z per sail section strip, y data for lift sails, y data for drag sails
    """
    cl = get_C(myvlm.panels, myvlm.areas, myvlm.lift, myvlm.inlet_conditions.V_app_infs, myvlm.n_spanwise, myvlm.n_chordwise, myvlm.rho)
    cd = get_C(myvlm.panels, myvlm.areas, myvlm.drag, myvlm.inlet_conditions.V_app_infs, myvlm.n_spanwise, myvlm.n_chordwise, myvlm.rho)
    
    k = int(myvlm.panels.shape[0] / (solver.n_spanwise * solver.n_chordwise * 2) ) # *2 in denominator due to underwater part

    sail_areas = np.split(myvlm.areas, (2*k))
    section_areas = np.zeros((k, solver.n_spanwise), dtype=float) # reshaped in proper way array with areas
    sail_cps = np.split(myvlm.cp, (2*k))
    sail_section_mean_cp_z = np.zeros((k, solver.n_spanwise), dtype=float)
    y1_data_list = []
    y2_data_list = []
    # cl for every panel in all sails
    CLs = np.zeros((k, 1), dtype=float) # array with CL per sail
    CDs = np.zeros((k, 1), dtype=float) # array with CL per sail
    # iterating only above water
    for i in range(k):
        for j in range(solver.n_spanwise):   
            section_sail_areas = sail_areas[i].reshape(solver.n_chordwise, solver.n_spanwise)[:,j]
            section_areas[i, j] = np.sum(section_sail_areas)

            arr = sail_cps[i][:, 2].reshape(solver.n_chordwise, solver.n_spanwise).transpose()
            mean_section_cp_z = np.sum(arr, axis=1) / solver.n_chordwise
            sail_section_mean_cp_z[i] = mean_section_cp_z

        CLs[i] = np.sum(section_areas[i] * cl[i]) / np.sum(section_areas[i])
        CDs[i] = np.sum(section_areas[i] * cd[i]) / np.sum(section_areas[i])
        y1_data_list.append(cl[i] / CLs[i])
        y2_data_list.append(cd[i] / CDs[i])
        
    return sail_section_mean_cp_z, y1_data_list, y2_data_list