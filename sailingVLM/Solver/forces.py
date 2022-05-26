import numpy as np
from sailingVLM.Solver.vlm_solver import calc_induced_velocity


def calc_force_inviscid_xyz(V_app_fs_at_cp, gamma_magnitude, span_vectors, rho):
    N = len(gamma_magnitude)
    force_xyz = np.full((N, 3), 0., dtype=float)
    for i in range(0, N):
        gamma = span_vectors[i] * gamma_magnitude[i]
        force_xyz[i] = rho * np.cross(V_app_fs_at_cp[i], gamma)

    return force_xyz


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


def extract_above_water_quantities(quantities, cp_points):
    # potential pitfalls
    # 1) above_water_mask = Mx < 0  # don't do this - consider a negative camber design ;p
    # 2) total_force = np.sum(force, axis=0) / 2.  # don't do this - the z component of the force would vanish due to mirror effect
    above_water_mask = cp_points[:, 2] > 0
    above_water_quantities = quantities[above_water_mask]
    total_above_water_quantities = np.sum(above_water_quantities, axis=0)  # heeling, sway, yaw (z-axis)
    return above_water_quantities, total_above_water_quantities


def calc_V_at_cp(V_app_infw, gamma_magnitude, panels):
    """
    :param V_app_infw: apparent wind
    :param gamma_magnitude: vector with circulations
    :param panels:
    :return: Wind at cp = apparent wind + wind_induced_at_cp
    """
    panels_1d = panels.flatten()
    N = len(panels_1d)
    v_ind_coeff = np.full((N, N, 3), 0., dtype=float)

    for i in range(0, N):
        cp = panels_1d[i].get_cp_position()
        for j in range(0, N):
            # velocity induced at i-th control point by j-th vortex
            v_ind_coeff[i][j] = panels_1d[j].get_induced_velocity(cp, V_app_infw[j])

    V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
    V_at_cp = V_app_infw + V_induced
    return V_at_cp, V_induced


def calc_force_wrapper(V_app_infw, gamma_magnitude, panels, rho):
    """
    force = rho* (V_app_fw_at_cp x gamma)
    :param V_app_infw: apparent wind
    :param gamma_magnitude: vector with circulations
    :param rho: air density
    :return: force
    """
    panels_1d = panels.flatten()
    N = len(panels_1d)
    V_at_cp, V_induced = calc_V_at_cp(V_app_infw, gamma_magnitude, panels)

    force = np.full((N, 3), 0., dtype=float)
    for i in range(0, N):
        gamma = panels_1d[i].get_span_vector() * gamma_magnitude[i]  # this form is shorter
        force[i] = rho * np.cross(V_at_cp[i], gamma)

    return force

def calc_V_at_cp_new(V_app_infw, gamma_magnitude, panels):
    """
    :param V_app_infw: apparent wind
    :param gamma_magnitude: vector with circulations
    :param panels:
    :return: Wind at cp = apparent wind + wind_induced_at_cp
    """
    panels_1d = panels.flatten()
    N = len(panels_1d)
    v_ind_coeff = np.full((N, N, 3), 0., dtype=float)

    for i in range(0, N):
        cp = panels_1d[i].get_cp_position()
        for j in range(0, N):
            # velocity induced at i-th control point by j-th vortex
            v_ind_coeff[i][j] = panels_1d[j].get_induced_velocity(cp, V_app_infw[j])

    V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
    # V_induced_re = V_induced.reshape(N, 3)
    V_at_cp = V_app_infw + V_induced
    return V_at_cp, V_induced


def calc_force_wrapper_new(V_app_infw, gamma_magnitude, panels, rho):
    # Katz and Plotkin, p. 346 Chapter 12 / Three-Dimensional Numerical Solution
    # f. Secondary Computations: Pressures, Loads, Velocities, Etc
    #Eq (12.25)

    V_at_cp, V_induced = calc_V_at_cp_new(V_app_infw, gamma_magnitude, panels)

    V_at_cp_re = V_at_cp.reshape(panels.shape[0], panels.shape[1], 3)
    gamma_re = gamma_magnitude.reshape(panels.shape)
    force_re_xyz = np.full((panels.shape[0], panels.shape[1], 3), 0., dtype=float)

    for i in range(0, panels.shape[0]):
        for j in range(0, panels.shape[1]):
            if i == 0:  # leading edge only
                gamma = panels[i, j].get_span_vector() * gamma_re[i, j]
            else:
                gamma = panels[i, j].get_span_vector() * (gamma_re[i, j]-gamma_re[i-1, j])

            force_re_xyz[i, j, :] = rho * np.cross(V_at_cp_re[i, j], gamma)

    return force_re_xyz


def  calc_pressure(force, panels):
    panels_1d = panels.flatten()
    n = len(panels_1d)
    p = np.zeros(shape=n)

    for i in range(n):
        area = panels_1d[i].get_panel_area()
        n = panels_1d[i].get_normal_to_panel()
        p[i] = np.dot(force[i], n) / area

    return p


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


