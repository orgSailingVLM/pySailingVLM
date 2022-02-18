import numpy as np
from Solver.Panel import Panel
from Solver.trailingedgepanel import TrailingEdgePanel
# pomyslec nad jit by kod byl kompilowany
# to mozna zrownoleglic jakos
def assembly_sys_of_eq(V_app_infw, panels):
    # lista paneli jest wolna - zmiejszamy ilosc wymiarow listy
    # w liscie jest teraz 100 paneli z czego 20 ostatnich jest trailingpanels
    panels1D = panels.flatten()
    N = len(panels1D)

    A = np.zeros(shape=(N, N))  # Aerodynamic Influence Coefficient matrix
    RHS = np.zeros(shape=N)
    v_ind_coeff = np.full((N, N, 3), 0., dtype=float)
    # i = jeden watek
    # i - panel itrator
    # j - vortex iterator
    for i in range(0, N):
        panel_surf_normal = panels1D[i].get_normal_to_panel()
        ctr_p = panels1D[i].get_ctr_point_position()
        RHS[i] = -np.dot(V_app_infw[i], panel_surf_normal)
        #print(type(panels1D[i]))

        for j in range(0, N):

                # velocity induced at i-th control point by j-th vortex
                # tutaj mozna pomyslec zeby wywolac get_horse_shoe_induced_velocity lub get_vortex ... znajdujace sie w Panelu
                # poza tym funkcja po kropce robi obliczenia cross, mozna cos zrobic py ta funkcja teraz sie tu liczyla szybciej
                # na tab;icy zawierajacej linie pedu
                #if j == N-1:
                   # print("tutaj N=", N)
                    #v_ind_coeff[i][j] = panels1D[j].get_horse_shoe_induced_velocity(ctr_p, V_app_infw[j])
                #else:
                #    v_ind_coeff[i][j] = panels1D[j].get_vortex_ring_induced_velocity()
                # tak bylo
                if isinstance(panels1D[i], TrailingEdgePanel):
                    #print("ok")
                    v_ind_coeff[i][j] = panels1D[j].get_horse_shoe_induced_velocity(ctr_p, V_app_infw[j])
                else:
                    v_ind_coeff[i][j] = panels1D[j].get_horse_shoe_induced_velocity(ctr_p, V_app_infw[j])
                    #v_ind_coeff[i][j] = panels1D[j].get_vortex_ring_induced_velocity()
                    #test = panels1D[j].get_vortex_ring_induced_velocity()
                    #print("DEZD: ", np.shape(test), "zawartosc: ", test)
                #v_ind_coeff[i][j] = panels1D[j].get_induced_velocity(ctr_p, V_app_infw[j])
                A[i][j] = np.dot(v_ind_coeff[i][j], panel_surf_normal)

    return A, RHS, v_ind_coeff  # np.array(v_ind_coeff)


def calc_circulation(V_app_ifnw, panels):
    # it is assumed that the freestream velocity is V [vx,0,vz], where vx > 0

    A, RHS, v_ind_coeff = assembly_sys_of_eq(V_app_ifnw, panels)
    print(A)
    print(RHS)
    print(np.shape(A), " ", np.shape(RHS))
    gamma_magnitude = np.linalg.solve(A, RHS)

    return gamma_magnitude, v_ind_coeff


def calc_induced_velocity(v_ind_coeff, gamma_magnitude):
    N = len(gamma_magnitude)
    V_induced = np.full((N, 3), 0., dtype=float)
    for i in range(N):
        for j in range(N):
            V_induced[i] += v_ind_coeff[i][j] * gamma_magnitude[j]

    return V_induced


def is_no_flux_BC_satisfied(V_app_fw, panels):
    panels1D = panels.flatten()
    N = len(panels1D)
    flux_through_panel = np.zeros(shape=N)
    panels_area = np.zeros(shape=N)

    for i in range(0, N):
        panel_surf_normal = panels1D[i].get_normal_to_panel()
        panels_area[i] = panels1D[i].get_panel_area()
        flux_through_panel[i] = -np.dot(V_app_fw[i], panel_surf_normal)

    for area in panels_area:
        if np.isnan(area) or area < 1E-14:
            raise ValueError("Solution error, panel_area is suspicious")

    for flux in flux_through_panel:
        if abs(flux) > 1E-12 or np.isnan(flux):
            raise ValueError("Solution error, there shall be no flow through panel!")

    return True


def calculate_app_fs(inletConditions, v_ind_coeff, gamma_magnitude):
    V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
    V_app_fs = inletConditions.V_app_infs + V_induced
    return V_induced, V_app_fs
