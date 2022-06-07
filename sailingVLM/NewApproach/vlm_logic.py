import numpy as np
from typing import Tuple
from numpy.linalg import norm

from sailingVLM.Solver.mesher import discrete_segment, make_point_mesh
from sailingVLM.Rotations.geometry_calc import rotation_matrix

def get_leading_edge_mid_point(p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    return (p2 + p3) / 2.

def get_trailing_edge_mid_points(p1: np.ndarray, p4: np.ndarray) -> np.ndarray:
    return (p4 + p1) / 2.

def calculate_normals_collocations_cps_rings_spans(panels: np.ndarray, gamma_orientation : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = panels.shape[0]
    ns = np.zeros((K, 3))
    span_vectors = np.zeros((K, 3))
    collocation_points = np.zeros((K, 3))
    center_of_pressure = np.zeros((K, 3))
    rings = np.zeros(shape=panels.shape)
    for idx, panel in enumerate(panels):
        p1 = panel[0]
        p2 = panel[1]
        p3 = panel[2]
        p4 = panel[3]

        vect_A = p4 - p2
        vect_B = p3 - p1

        leading_mid_point = get_leading_edge_mid_point(p2, p3)
        trailing_edge_mid_point = get_trailing_edge_mid_points(p1, p4)
        dist = trailing_edge_mid_point - leading_mid_point

        collocation_points[idx] = leading_mid_point + 0.75 * dist
        center_of_pressure[idx] = leading_mid_point + 0.25 * dist

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
    return ns, collocation_points, center_of_pressure, rings, span_vectors
 
def is_in_vortex_core(vector_list):
    for vec in vector_list:
        if norm(vec) < 1e-9:
            return True

def vortex_line(p: np.array, p1: np.array, p2: np.array, gamma: float = 1.0) -> np.array:
    # strona 254
    r0 = np.array(p2 - p1)
    r1 = np.array(p - p1)
    r2 = np.array(p - p2)

    r1_cross_r2 = np.cross(r1, r2)
    
    q_ind = np.array([0, 0, 0])
    if is_in_vortex_core([r1, r2, r1_cross_r2]):
        return [0.0, 0.0, 0.0]
    else:
        q_ind = r1_cross_r2 / np.square(np.linalg.norm(r1_cross_r2))
        q_ind *= np.dot(r0, (r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2)))
        q_ind *= gamma / (4 * np.pi)

    return q_ind

def vortex_infinite_line(P: np.ndarray, A: np.array, r0: np.ndarray, gamma : float = 1.0):

    u_inf = r0 / norm(r0)
    ap = P - A
    norm_ap = norm(ap)

    v_ind = np.cross(u_inf, ap) / (
                norm_ap * (norm_ap - np.dot(u_inf, ap)))  # todo: consider checking is_in_vortex_core
    v_ind *= gamma / (4. * np.pi)
    return v_ind

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

def vortex_ring(p: np.array, A: np.array, B: np.array, C: np.array, D: np.array,
                gamma: float = 1.0) -> np.array:

    sub1 = vortex_line(p, A, B, gamma)
    sub2 = vortex_line(p, B, C, gamma)
    sub3 = vortex_line(p, C, D, gamma)
    sub4 = vortex_line(p, D, A, gamma)

    q_ind = sub1 + sub2 + sub3 + sub4
    return q_ind

def get_influence_coefficients_spanwise(collocation_points: np.ndarray, rings: np.ndarray, normals: np.ndarray, M: int, N: int, V_app_infw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    m = collocation_points.shape[0]

    RHS = [-np.dot(V_app_infw[i], normals[i]) for i in range(normals.shape[0])]
    coefs = np.zeros((m, m))
    wind_coefs = np.zeros((m, m, 3))
    trailing_rings = []
    # loop over other vortices
    for i, ring in enumerate(rings):
        A = ring[0]
        B = ring[1]
        C = ring[2]
        D = ring[3]
        # loop over points
        for j, point in enumerate(collocation_points):
           
            a = vortex_ring(point, A, B, C, D)
            # poprawka na trailing edge
            # todo: zrobic to w drugim, oddzielnym ifie
            if i >= len(collocation_points) - M:
                #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                a = vortex_horseshoe(point, ring[1], ring[2], V_app_infw[i])
            b = np.dot(a, normals[j].reshape(3, 1))
            wind_coefs[j, i] = a
            coefs[j, i] = b
    RHS = np.asarray(RHS)
    
    for j, ring in enumerate(rings):
        if j >= len(collocation_points) - M:
            A = ring[0]
            B = ring[1]
            C = ring[2]
            D = ring[3]
            trailing_rings.append([A, B, C, D])
                
    return coefs, RHS, wind_coefs, trailing_rings

def solve_eq(coefs: np.ndarray, RHS: np.ndarray):
    big_gamma = np.linalg.solve(coefs, RHS)
    return big_gamma

def calc_induced_velocity(v_ind_coeff, gamma_magnitude):
    N = gamma_magnitude.shape[0]
    
    t2 = len(gamma_magnitude)
    V_induced = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            V_induced[i] += v_ind_coeff[i,j] * gamma_magnitude[j]

    return V_induced


def get_panels_area(panels: np.ndarray, N: int, M: int)-> np.ndarray:
    
    m = N * M
    areas = np.zeros(m, dtype=float)
    sh = panels.shape[0]
    for i in range(0, sh):
        
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
    flux_through_panel = np.zeros(shape=N)
    #panels_area = np.zeros(shape=N)

    # dla kazdego panelu
    for i in range(0, N):
        #panel_surf_normal = panels[i].get_normal_to_panel()
        #panels_area[i] = panels[i].get_panel_area()
        flux_through_panel[i] = -np.dot(V_app_fw[i], normals[i])

    for area in areas:
        if np.isnan(area) or area < 1E-14:
            raise ValueError("Solution error, panel_area is suspicious")

    for flux in flux_through_panel:
        if abs(flux) > 1E-12 or np.isnan(flux):
            raise ValueError("Solution error, there shall be no flow through panel!")

    return True

# czesc kodu sie powtarza, zrobic osobna funkcje
def calc_V_at_cp_new(V_app_infw, gamma_magnitude, panels, center_of_pressure, rings, M, N, normals):
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
                a = vortex_ring(point, A, B, C, D)

                # poprawka na trailing edge
                # todo: zrobic to w drugim, oddzielnym ifie
                if j >= len(center_of_pressure) - M:
                    #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                    a = vortex_horseshoe(point, ring[1], ring[2], V_app_infw[j])
                b = np.dot(a, normals[i].reshape(3, 1))
                # v_ind_coeff to jest u mnie wind_coefs
                wind_coefs[i, j] = a
                coefs[i, j] = b

        V_induced = calc_induced_velocity(wind_coefs, gamma_magnitude)
        V_at_cp = V_app_infw + V_induced
        return V_at_cp, V_induced


def calc_force_wrapper_new(V_app_infw, gamma_magnitude, panels, rho, center_of_pressure, rings, M, N, normals, span_vectors):
    # Katz and Plotkin, p. 346 Chapter 12 / Three-Dimensional Numerical Solution
    # f. Secondary Computations: Pressures, Loads, Velocities, Etc
    #Eq (12.25)

    V_at_cp, V_induced = calc_V_at_cp_new(V_app_infw, gamma_magnitude, panels, center_of_pressure, rings, M, N, normals)

    K = M * N
    force_xyz = np.zeros((K, 3))

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


def calc_pressure(forces, normals, areas, N , M):
    K = N*M
    p = np.zeros(shape=K)

    for i in range(K):
        p[i] = np.dot(forces[i], normals[i]) / areas[i]

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
