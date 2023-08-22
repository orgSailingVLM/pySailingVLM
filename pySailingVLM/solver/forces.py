import numpy as np
import sys
from typing import Tuple
from pySailingVLM.solver.coefs import calc_velocity_coefs
from pySailingVLM.solver.velocity import calculate_app_fs

def calc_moment_arm_in_shifted_csys(cp_points, v_from_old_2_new_csys):
    dx, dy, dz = v_from_old_2_new_csys
    N = len(cp_points)
    r = np.copy(cp_points)
    for i in range(N):
        r[i, 0] -= dx  # yawing moment arm
        r[i, 1] -= dy  # pitching moment arm
        if cp_points[i, 2] > 0:
            r[i, 2] -= dz  # heeling moment arm
        else:
            r[i, 2] += dz

    return r

def calc_moments(cp_points, forces):
    moments = np.cross(cp_points, forces)
    # Mx_test1 = cp_points[:, 2] * force_xyz[:, 1]
    # Mx_test2 = cp_points[:, 1] * force_xyz[:, 2]
    # Mx_test = Mx_test1 + Mx_test2
    return moments

def determine_vector_from_its_dot_and_cross_product(F, r_dot_F, r_cross_F):
    # https://math.stackexchange.com/questions/246594/what-is-vector-division
    # https://math.stackexchange.com/questions/1683996/determining-an-unknown-vector-from-its-cross-and-dot-product-with-known-vector
    # we know that: a x ( b x c) = b(a . c) - c(a.b)
    # so: F x ( r x F) = r(F . F) - F(F.r)
    # one shall get r, assuming that both the cross and dot product are known:
    # r = (F x (r x F) + F(F . r))/(F . F)

    F_dot_F = np.dot(F, F)
    R = (np.cross(F, r_cross_F) + F * r_dot_F) / F_dot_F
    return np.array(R)




# to bylo w starej finkcji:
# # "Loop serialization occurs when any number of prange driven loops are present 
# # inside another prange driven loop. In this case the outermost of all the prange loops executes
# # in parallel and any inner prange loops (nested or otherwise) 
# # are treated as standard range based loops. Essentially, nested parallelism does not occur."
# #@numba.jit(nopython=True, parallel=True)

# cp - center of pressure
def calc_force_wrapper(V_app_infw: np.ndarray, gamma_magnitude: np.ndarray, rho: float, cp: np.ndarray, rings: np.ndarray, n_spanwise: int, normals: np.ndarray, span_vectors: np.ndarray, trailing_edge_info : np.ndarray, leading_edges_info : np.ndarray, force_name: str = 'force_xyz', gamma_orientation : float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    calc_force_wrapper calculate force

    :param np.ndarray V_app_infw: apparent wind velocity for an infinite sail (without induced wind velocity)
    :param np.ndarray gamma_magnitude: gamma magnitude
    :param float rho: rho
    :param np.ndarray cp: center of pressure
    :param np.ndarray rings: _description_
    :param int n_spanwise: _description_
    :param np.ndarray normals: _description_
    :param np.ndarray span_vectors: _description_
    :param np.ndarray trailing_edge_info: _description_
    :param np.ndarray leading_edges_info: _description_
    :param str force_name: name of force to be calculated, defaults to 'force_xyz', possible 'lift', 'drag', 'force_xyz'
    :param float gamma_orientation: ogrientation of gamma, defaults to 1.0
    :return Tuple[np.ndarray, np.ndarray, np.ndarray]: return calculated force, V_at_cp and V_induced
    """
    
    # check if force_name is allowed
    try:
        force_name_allowed = ('lift', 'drag', 'force_xyz')
        if not force_name in force_name_allowed:
            raise ValueError(f'Force name is not allowed. You can use one of these: {force_name_allowed}')
    except ValueError as e:
        print(e)
        sys.exit()
    
    # Katz and Plotkin, p. 346 Chapter 12 / Three-Dimensional Numerical Solution
    # f. Secondary Computations: Pressures, Loads, Velocities, Etc
    # Eq (12.25)
    _, v_ind_coeff = calc_velocity_coefs(V_app_infw, cp, rings, normals, trailing_edge_info, gamma_orientation)
    V_induced, V_at_cp = calculate_app_fs(V_app_infw, v_ind_coeff, gamma_magnitude)
    
    
    V_for_calculations = V_at_cp    
    if force_name == 'lift':
        V_for_calculations = V_app_infw
    elif force_name == 'drag':
        V_for_calculations = V_induced
    
    # if case 1x1 leading_edges_info is False False False False
    # horseshoe_edge_info i True True True True
    # caclulating winds as for trailing edges
    # forces and pressure like "leading edge"
    case1x1 = np.logical_not(np.any(leading_edges_info)) 
    
    K = cp.shape[0]
    force = np.zeros((K, 3))
    #numba.prange
    for i in range(K):
        # for spanwise only!
        # if panel is leading edge
        gamma = 0.0
        if leading_edges_info[i] or case1x1:
            gamma = span_vectors[i] * gamma_magnitude[i]
        else:
            gamma = span_vectors[i] * (gamma_magnitude[i] - gamma_magnitude[i-n_spanwise])
        force[i] = rho * np.cross(V_for_calculations[i], gamma)
        
    return force, V_at_cp, V_induced


def calc_pressure(forces, normals, areas):
    p = forces.dot(normals.transpose()).diagonal() /  areas
    return p


def is_no_flux_BC_satisfied(V_app_fw, panels, areas, normals):

    flux_through_panel = -V_app_fw.dot(normals.transpose()).diagonal()

    for area in areas:
        if np.isnan(area) or area < 1E-14:
            raise ValueError("Solution error, panel_area is suspicious")

    for flux in flux_through_panel:
        if abs(flux) > 1E-12 or np.isnan(flux):
            raise ValueError("Solution error, there shall be no flow through panel!")

    return True

def calc_pressure_coeff(pressure: np.ndarray, rho: float, V: np.ndarray) -> np.ndarray:
    """
    calc_pressure_coeff calculate pressure coeffs

    :param np.ndarray pressure: array with pressure
    :param float rho: liquid density
    :param np.ndarray V: free stream speed
    :return np.ndarray: pressure coeffs
    """
    # 
    p_coeffs = np.array([ p / (0.5 * rho * np.dot(V[i], V[i])) for i, p in enumerate(pressure)])
    return p_coeffs
    