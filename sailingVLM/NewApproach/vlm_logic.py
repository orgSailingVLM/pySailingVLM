import numpy as np
import pandas as pd
from typing import List, Tuple
from numpy.linalg import norm

from sailingVLM.Solver.mesher import discrete_segment, make_point_mesh
from sailingVLM.Rotations.geometry_calc import rotation_matrix

import numba

from sailingVLM.YachtGeometry.SailGeometry import SailGeometry, SailSet
from sailingVLM.Rotations.CSYS_transformations import CSYS_transformations


def get_leading_edge_mid_point(p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    return (p2 + p3) / 2.

def get_trailing_edge_mid_points(p1: np.ndarray, p4: np.ndarray) -> np.ndarray:
    return (p4 + p1) / 2.

def calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels: np.ndarray, gamma_orientation : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        #####
            
        n = np.cross(vect_A, vect_B)
        n = n / np.linalg.norm(n)
        ns[idx] = n
    return ns, collocation_points, center_of_pressure, rings, span_vectors, leading_mid_points, trailing_edge_mid_points

@numba.jit(nopython=True, cache=True)
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
@numba.jit(numba.float64[::1](numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.optional(numba.float64)), nopython=True, debug = True, cache=True) 
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
@numba.jit(nopython=True, cache=True)
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

@numba.jit(nopython=True, cache=True)
def vortex_ring(p: np.array, A: np.array, B: np.array, C: np.array, D: np.array,
                gamma: float = 1.0) -> np.array:

    sub1 = vortex_line(p, A, B, gamma)
    #assert not vortex_line.nopython_signatures
    sub2 = vortex_line(p, B, C, gamma)
    sub3 = vortex_line(p, C, D, gamma)
    sub4 = vortex_line(p, D, A, gamma)

    q_ind = sub1 + sub2 + sub3 + sub4
    return q_ind

# sails = [jib, main]
def get_influence_coefficients_spanwise_jib_version(collocation_points: np.ndarray, rings: np.ndarray, normals: np.ndarray, V_app_infw: np.ndarray, sails : List[SailGeometry], horseshoe_info : np.ndarray, gamma_orientation : float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
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
           
            a = vortex_ring(point, A, B, C, D, gamma_orientation)
            # poprawka na trailing edge
            # todo: zrobic to w drugim, oddzielnym ifie
            # poziomo od 0 do n-1, reszta odzielnie
            if horseshoe_info[i]:
                #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                a = vortex_horseshoe(point, ring[1], ring[2], V_app_infw[i], gamma_orientation)
            b = np.dot(a, normals[j].reshape(3, 1))
            wind_coefs[j, i] = a
            coefs[j, i] = b
    RHS = np.asarray(RHS)
                
    return coefs, RHS, wind_coefs

# numba tutaj nie rozumie typow -> do poprawki
#@numba.jit(nopython=True)
#@numba.njit(parallel=True)
def get_influence_coefficients_spanwise(collocation_points: np.ndarray, rings: np.ndarray, normals: np.ndarray, M: int, N: int, V_app_infw: np.ndarray, gamma_orientation : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    m = collocation_points.shape[0]
    RHS = -V_app_infw.dot(normals.transpose()).diagonal()
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
           
            a = vortex_ring(point, A, B, C, D, gamma_orientation)
            # poprawka na trailing edge
            # todo: zrobic to w drugim, oddzielnym ifie
            # poziomo od 0 do n-1, reszta odzielnie
            if i >= len(collocation_points) - M:
                #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                a = vortex_horseshoe(point, ring[1], ring[2], V_app_infw[i], gamma_orientation)
            b = np.dot(a, normals[j].reshape(3, 1))
            wind_coefs[j, i] = a
            coefs[j, i] = b
    RHS = np.asarray(RHS)
                
    return coefs, RHS, wind_coefs

def solve_eq(coefs: np.ndarray, RHS: np.ndarray):
    big_gamma = np.linalg.solve(coefs, RHS)
    return big_gamma

# parallel - no time change
@numba.jit(nopython=True, cache=True)
def calc_induced_velocity(v_ind_coeff, gamma_magnitude):
    N = gamma_magnitude.shape[0]
    
    t2 = len(gamma_magnitude)
    V_induced = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            V_induced[i] += v_ind_coeff[i,j] * gamma_magnitude[j]

    return V_induced

# no seppedup
#@numba.jit(nopython=True, parallel=True)
def get_panels_area(panels: np.ndarray)-> np.ndarray:
    
    #m = N * M
    m = panels.shape[0]
    areas = np.zeros(m)
    #areas = np.zeros(m, dtype=float)
    sh = panels.shape[0]
    # numba.prange
    # range works slightly quicker than numba.prange (without decorator of course)
    for i in range(sh):
        
        p = [panels[i, 0], panels[i, 1], panels[i, 2], panels[i, 3]]
        path = []
        for j in range(len(p) - 1):
            step = p[j + 1] - p[j]
            path.append(step)

        area = 0
        for k in range(len(path) - 1):
            s = np.cross(path[k], path[k + 1])
            s = np.linalg.norm(s)
            area += 0.5 * s
        areas[i] = area   

    return areas

def is_no_flux_BC_satisfied(V_app_fw, panels, areas, normals):

    N = panels.shape[0]

    flux_through_panel = -V_app_fw.dot(normals.transpose()).diagonal()

    for area in areas:
        if np.isnan(area) or area < 1E-14:
            raise ValueError("Solution error, panel_area is suspicious")

    for flux in flux_through_panel:
        if abs(flux) > 1E-12 or np.isnan(flux):
            raise ValueError("Solution error, there shall be no flow through panel!")

    return True


def calc_V_at_cp_new_jib_version(V_app_infw, gamma_magnitude, center_of_pressure, rings, N, normals, sails : List[SailGeometry], trailing_edge_info : np.ndarray, gamma_orientation : np.ndarray):
        
    
    m = center_of_pressure.shape[0]

    coefs = np.zeros((m, m))
    wind_coefs = np.zeros((m, m, 3))
    for i, point in enumerate(center_of_pressure):

        # loop over other vortices
        for j, ring in enumerate(rings):
            A = ring[0]
            B = ring[1]
            C = ring[2]
            D = ring[3]
            a = vortex_ring(point, A, B, C, D, gamma_orientation)

            # poprawka na trailing edge
            # todo: zrobic to w drugim, oddzielnym ifie
            if trailing_edge_info[j]:
                #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                a = vortex_horseshoe(point, ring[1], ring[2], V_app_infw[j], gamma_orientation)
            b = np.dot(a, normals[i].reshape(3, 1))
            # v_ind_coeff to jest u mnie wind_coefs
            wind_coefs[i, j] = a
            coefs[i, j] = b

    V_induced = calc_induced_velocity(wind_coefs, gamma_magnitude)
    V_at_cp = V_app_infw + V_induced
    return V_at_cp, V_induced

    
    
# czesc kodu sie powtarza, zrobic osobna funkcje
# todo numba tutaj nie rozumie typow
#@numba.jit(nopython=True)
def calc_V_at_cp_new(V_app_infw, gamma_magnitude, panels, center_of_pressure, rings, M, N, normals, gamma_orientation : np.ndarray):
        m = M * N
        coefs = np.zeros((m, m))
        wind_coefs = np.zeros((m, m, 3))
        for i, point in enumerate(center_of_pressure):

            # loop over other vortices
            for j, ring in enumerate(rings):
                A = ring[0]
                B = ring[1]
                C = ring[2]
                D = ring[3]
                a = vortex_ring(point, A, B, C, D, gamma_orientation)

                # poprawka na trailing edge
                # todo: zrobic to w drugim, oddzielnym ifie
                if j >= len(center_of_pressure) - M:
                    #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                    a = vortex_horseshoe(point, ring[1], ring[2], V_app_infw[j], gamma_orientation)
                b = np.dot(a, normals[i].reshape(3, 1))
                # v_ind_coeff to jest u mnie wind_coefs
                wind_coefs[i, j] = a
                coefs[i, j] = b

        V_induced = calc_induced_velocity(wind_coefs, gamma_magnitude)
        V_at_cp = V_app_infw + V_induced
        return V_at_cp, V_induced

# "Loop serialization occurs when any number of prange driven loops are present 
# inside another prange driven loop. In this case the outermost of all the prange loops executes
# in parallel and any inner prange loops (nested or otherwise) 
# are treated as standard range based loops. Essentially, nested parallelism does not occur."
#@numba.jit(nopython=True, parallel=True)
def calc_force_wrapper_new(V_app_infw, gamma_magnitude, panels, rho, center_of_pressure, rings, M, N, normals, span_vectors):
    # Katz and Plotkin, p. 346 Chapter 12 / Three-Dimensional Numerical Solution
    # f. Secondary Computations: Pressures, Loads, Velocities, Etc
    #Eq (12.25)

    V_at_cp, V_induced = calc_V_at_cp_new(V_app_infw, gamma_magnitude, panels, center_of_pressure, rings, M, N, normals)

    K = M * N
    force_xyz = np.zeros((K, 3))
    #numba.prange
    for i in range(K):
        # for spanwise only!
        # if panel is leading edge
        gamma = 0.0
        if i < M:
            gamma = span_vectors[i] * gamma_magnitude[i]
        else:
            gamma = span_vectors[i] * (gamma_magnitude[i] - gamma_magnitude[i-M])
        force_xyz[i] = rho * np.cross(V_at_cp[i], gamma)

    return force_xyz


def calc_force_wrapper_new_jib_version(V_app_infw, gamma_magnitude, rho, center_of_pressure, rings, M, N, normals, span_vectors, sails :List[SailGeometry], trailing_edge_info : np.ndarray, leading_edges_info : np.ndarray, gamma_orientation : float = 1.0):
    # Katz and Plotkin, p. 346 Chapter 12 / Three-Dimensional Numerical Solution
    # f. Secondary Computations: Pressures, Loads, Velocities, Etc
    #Eq (12.25)
    ##### WAZNE #####
    # N - odleglosc miedzy leading a trailing edge
    # M - rozpietosc skrzydel    
    V_at_cp, V_induced = calc_V_at_cp_new_jib_version(V_app_infw, gamma_magnitude, center_of_pressure, rings, N, normals, sails, trailing_edge_info, gamma_orientation)
    
    # if case 1x1 leading_edges_info is False False False False
    # horseshoe_edge_info i True True True True
    # caclulating winds as for trailing edges
    # forces and pressure like "leading edge"
    case1x1 = np.logical_not(np.any(leading_edges_info)) 
    
    K = center_of_pressure.shape[0]
    force_xyz = np.zeros((K, 3))
    #numba.prange
    for i in range(K):
        # for spanwise only!
        # if panel is leading edge
        gamma = 0.0
        if leading_edges_info[i] or case1x1:
            gamma = span_vectors[i] * gamma_magnitude[i]
        else:
            gamma = span_vectors[i] * (gamma_magnitude[i] - gamma_magnitude[i-M])
        force_xyz[i] = rho * np.cross(V_at_cp[i], gamma)
    return force_xyz, V_at_cp, V_induced




def calc_pressure_new_approach(forces, normals, areas, N , M):
    p = forces.dot(normals.transpose()).diagonal() /  areas
    return p


def get_vlm_CL_CD_free_wing(F: np.ndarray, V: np.array, rho : float, S : float) -> Tuple[float, float]:
    
    total_F = np.sum(F, axis=0)
    q = 0.5 * rho * (np.linalg.norm(V) ** 2) * S
    CL_vlm = total_F[2] / q
    CD_vlm = total_F[0] / q
    
    return CL_vlm, CD_vlm

################ mesher #################
# sprawdzic typy!


def make_panels_from_mesh_spanwise_new(mesh, gamma_orientation : float) -> np.array:
    n_lines = mesh.shape[0]
    n_points_per_line = mesh.shape[1]
    M = n_points_per_line - 1
    N = n_lines - 1
    
    new_approach_panels = np.zeros((N * M, 4, 3))

    counter = 0
    for i in range(N):
        for j in range(M):
            pSE = mesh[i + 1][j]
            pSW = mesh[i][j]
            pNW = mesh[i][j + 1]
            pNE = mesh[i + 1][j + 1]
            new_approach_panels[counter] = [pSE, pSW, pNW, pNE] 

            counter += 1
    return  new_approach_panels



def make_panels_from_le_te_points_new(points, grid_size, gamma_orientation):
    """
    this is the main meshing method
    :param points: 
    :param grid_size: 
    :return: 
    """
    le_SW, te_SE, le_NW, te_NE = points
    nc, ns = grid_size
    south_line = discrete_segment(le_SW, te_SE, nc)
    north_line = discrete_segment(le_NW, te_NE, nc)

    mesh = make_point_mesh(south_line, north_line, ns)
    new_approach_panels = make_panels_from_mesh_spanwise_new(mesh, gamma_orientation)
    return new_approach_panels


def create_panels(half_wing_span : float, chord : float, AoA_deg : float, M : int, N : int) -> np.ndarray:
    le_NW = np.array([0., half_wing_span, 0.])      # leading edge North - West coordinate
    le_SW = np.array([0., -half_wing_span, 0.])     # leading edge South - West coordinate

    te_NE = np.array([chord, half_wing_span, 0.])   # trailing edge North - East coordinate
    te_SE = np.array([chord, -half_wing_span, 0.])  # trailing edge South - East coordinate

    Ry = rotation_matrix([0, 1, 0], np.deg2rad(AoA_deg))

    ### MESH DENSITY ###
    # sprawdzci czy nie jest odrotnie M i N
    ns = M    # number of panels (spanwise)
    nc = N   # number of panels (chordwise)


    new_approach_panels = make_panels_from_le_te_points_new(
        [np.dot(Ry, le_SW),
        np.dot(Ry, te_SE),
        np.dot(Ry, le_NW),
        np.dot(Ry, te_NE)],
        [nc, ns],
        gamma_orientation=1)
    return new_approach_panels


def extract_above_water_quantities_new_approach(quantities, cp_points):
    
    # for jib and main, quantities is always dividable by 2
    half = int(quantities.shape[0] / 2)
    above_water_quantities = quantities[0:half]
    total_above_water_quantities = np.sum(above_water_quantities, axis=0)  # heeling, sway, yaw (z-axis)
    return above_water_quantities, total_above_water_quantities



# to mozna gdzie s przenesc 
# było uprzednio w pliku sail geometry (208)

# def get_cp_straight_yacht(cp_points, csys_transformations):
#     cp_straight_yacht = np.array([csys_transformations.reverse_rotations_with_mirror(p) for p in cp_points])
#     return cp_straight_yacht

# # sail_cp_to_girths
# def get_y_as_girths(cp_points, csys_transformations, tack_mounting):
#     sail_cp_straight_yacht = get_cp_straight_yacht(cp_points, csys_transformations)
#     tack_mounting = tack_mounting
#     y = sail_cp_straight_yacht[:, 2]
#     y_as_girths = (y - tack_mounting[2]) / (max(y) - tack_mounting[2])
#     return y_as_girths

# # set
# def sail_cp_to_girths(self):
#         y_as_girths = np.array([])
#         for sail in self.sails:
#             y_as_girths = np.append(y_as_girths, sail.sail_cp_to_girths())
#         return y_as_girths

# # sail

# def sail_cp_to_girths(self):
#     sail_cp_straight_yacht = self.get_cp_points_upright()
#     tack_mounting = self.tack_mounting
#     y = sail_cp_straight_yacht[:, 2]
#     y_as_girths = (y - tack_mounting[2]) / (max(y) - tack_mounting[2])
#     return y_as_girths
    
# def get_cp_points_upright(cp_points):
#     cp_straight_yacht = np.array([self.csys_transformations.reverse_rotations_with_mirror(p) for p in cp_points])
#     return cp_straight_yacht


####

def get_cp_strainght_yacht(cp_points : np.ndarray, csys_transformations: CSYS_transformations) -> np.ndarray:
    """
    cp_strainght_yacht get center of pressure points straight to bridge

    :param np.ndarray cp_points: center of pressure points
    :return np.ndarray: staright center of pressure points
    """
    return np.array([csys_transformations.reverse_rotations_with_mirror(p) for p in cp_points])

def cp_to_girths(sail_cp_straight_yacht, tack_mounting):
    sail_cp_straight_yacht_z = sail_cp_straight_yacht[:, 2]
    z_as_girths = (sail_cp_straight_yacht_z - tack_mounting[2]) / (max(sail_cp_straight_yacht_z) - tack_mounting[2])
    return z_as_girths


def get_cp_z_as_girths_all(sail_set : SailSet, csys_transformations : CSYS_transformations, center_of_pressure : np.ndarray) -> np.ndarray: 
    """
    cp_z_as_girths_all get center of pressure staright z as girths

    :param SailSet sail_set: Sail set object
    :param CSYS_transformations csys_transformations: csys transformations
    :param np.ndarray center_of_pressure: array with all center of pressure points (for all sails and for above and under water)

    :return np.ndarray: y_as_girths for all sails (with above and under water points)
    """
    n = len(sail_set.sails)
    # 2* bo mamy odpicie lustrzane 
    chunks_cp_points = np.array_split(center_of_pressure, 2 * n)

    nxm = chunks_cp_points[0].shape[0]
    cp_z_as_girths_all = np.array([])
    #cp_straight_yacht_all = np.array([])
    
    cp_straight_yacht_all = np.empty((0,3))
    for i in range(n):
        sail_cp_points = np.concatenate([chunks_cp_points[i], chunks_cp_points[i+n]])
        sail_cp_straight_yacht = get_cp_strainght_yacht(sail_cp_points, csys_transformations)
        z_as_girths = cp_to_girths(sail_cp_straight_yacht, sail_set.sails[i].tack_mounting)
        cp_z_as_girths_all = np.append(cp_z_as_girths_all, z_as_girths)
        #cp_straight_yacht_all = np.append(cp_straight_yacht_all, sail_cp_straight_yacht)
        cp_straight_yacht_all = np.append(cp_straight_yacht_all, sail_cp_straight_yacht, axis=0)

    # ulozenie w array cp_z_as_girths_all :
    # jib above, jib under, main abobe, main under
    return cp_z_as_girths_all, cp_straight_yacht_all


# to policzyc dla wszystkich cpsow
# potem wziac gorna czesc czyli pierwsza polowe by miec nad woda

def get_cp_z_as_girths_all_above(cp_z_as_girths_all : np.ndarray, sail_set : SailSet):
    # 2 bo mamy odbicie 
    # parzyste numery to sa te nad woda
    n = 2 * len(sail_set.sails)
    z_splitted = np.split(cp_z_as_girths_all, n)
    # [::2] gets every second element from array z_splitted
    cp_z_as_girths_all_above = np.asarray(z_splitted[::2]).flatten()
    # cp_z_as_girths_all_above size
    repeat = int(cp_z_as_girths_all_above.shape[0] / 2)
    names = np.array([])
    
    for sail in sail_set.sails:
        rep_names = np.repeat(sail.name, repeat)
        names = np.append(names, rep_names)
      
    return cp_z_as_girths_all_above, names