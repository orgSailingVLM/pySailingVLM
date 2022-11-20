import numpy as np

from sailing_vlm.solver.coefs import calc_wind_coefs
from sailing_vlm.solver.velocity import calculate_app_fs

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

# import problem
# sails  :List[SailGeometry], 
def calc_force_wrapper(V_app_infw, gamma_magnitude, rho, center_of_pressure, rings, M, normals, span_vectors, trailing_edge_info : np.ndarray, leading_edges_info : np.ndarray, gamma_orientation : float = 1.0):
    # Katz and Plotkin, p. 346 Chapter 12 / Three-Dimensional Numerical Solution
    # f. Secondary Computations: Pressures, Loads, Velocities, Etc
    #Eq (12.25)
    ##### WAZNE #####
    # N - odleglosc miedzy leading a trailing edge
    # M - rozpietosc skrzydel    

    _, wind_coefs = calc_wind_coefs(V_app_infw, center_of_pressure, rings, normals, trailing_edge_info, gamma_orientation)
    V_induced, V_at_cp = calculate_app_fs(V_app_infw, wind_coefs, gamma_magnitude)

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



def calc_pressure(forces, normals, areas):
    p = forces.dot(normals.transpose()).diagonal() /  areas
    return p


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

