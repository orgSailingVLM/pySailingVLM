import timeit
import numpy as np


from pySailingVLM.rotations.geometry_calc import rotation_matrix
from pySailingVLM.solver.panels import make_panels_from_le_te_points, get_panels_area
from pySailingVLM.solver.coefs import get_CL_CD_free_wing, get_vlm_CL_CD_free_wing, get_vlm_Cxyz
from pySailingVLM.solver.coefs import calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points, \
                                solve_eq, calculate_RHS, calc_velocity_coefs

from pySailingVLM.solver.velocity import calculate_app_fs 
from pySailingVLM.solver.forces import is_no_flux_BC_satisfied, calc_force_wrapper, calc_pressure

class Aircraft:
    def __init__(self, chord_length : float, half_wing_span: float, AoA_deg : float, n_spanwise : int, n_chordwise, V : np.array, rho : float = 1.225, gamma_orientation = 1.0) -> None:
        self.chord_length = chord_length
        self.half_wing_span = half_wing_span
        self.AoA_deg = AoA_deg
        self.n_spanwise = n_spanwise
        self.n_chordwise = n_chordwise
        self.V = V
        self.rho = rho
        self.gamma_orientation = gamma_orientation

        
        
        self.__AR = 2 * self.half_wing_span / self.chord_length # aspect ratio
        self.__S = 2 * self.half_wing_span * self.chord_length
        
        self.__CL_theoretical, self.__CD_theoretical = get_CL_CD_free_wing(self.__AR, self.AoA_deg)

        
        # le_NW = np.array([0., self.half_wing_span, 0.])      # leading edge North - West coordinate
        # le_SW = np.array([0., -self.half_wing_span, 0.])     # leading edge South - West coordinate

        # te_NE = np.array([self.chord_length, self.half_wing_span, 0.])   # trailing edge North - East coordinate
        # te_SE = np.array([self.chord_length, -self.half_wing_span, 0.])  # trailing edge South - East coordinate
        
        # Ry = rotation_matrix([0, 1, 0], np.deg2rad(self.AoA_deg))
        
        le_NW = np.array([0., 0 , self.half_wing_span])      # leading edge North - West coordinate
        le_SW = np.array([0., 0., -self.half_wing_span])     # leading edge South - West coordinate

        te_NE = np.array([self.chord_length, 0., self.half_wing_span])   # trailing edge North - East coordinate
        te_SE = np.array([self.chord_length, 0., -self.half_wing_span])  # trailing edge South - East coordinate
        
        Rz = rotation_matrix([0, 0, 1], np.deg2rad(-self.AoA_deg))

        panels, trailing_edge_info, leading_edge_info = make_panels_from_le_te_points([np.dot(Rz, le_SW),
                                                                                    np.dot(Rz, te_SE),
                                                                                    np.dot(Rz, le_NW),
                                                                                    np.dot(Rz, te_NE)],
                                                                                    [self.n_chordwise, self.n_spanwise])
        
        N = panels.shape[0]
        self.__V_app_infw = np.array([self.V for i in range(N)])
        
        areas = get_panels_area(panels)
        normals, ctr_p, cp, rings, span_vectors, _, _ = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels, self.gamma_orientation)
        coefs, v_ind_coeff = calc_velocity_coefs(self.__V_app_infw, ctr_p, rings, normals, trailing_edge_info, self.gamma_orientation)
        self.__RHS = calculate_RHS(self.__V_app_infw, normals)
        self.__gamma_magnitude = solve_eq(coefs, self.__RHS)
        
        _,  V_app_fs_at_ctrl_p = calculate_app_fs(self.__V_app_infw,  v_ind_coeff,  self.__gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, panels, areas, normals)
        self.__force, _, _ = calc_force_wrapper(self.__V_app_infw, self.__gamma_magnitude, self.rho, cp, rings, self.n_spanwise, normals, span_vectors, trailing_edge_info, leading_edge_info, 'force_xyz', self.gamma_orientation)
        self.__pressure = calc_pressure(self.__force, normals, areas)

        self.__CL, self.__CD = get_vlm_CL_CD_free_wing(self.__force, self.V, self.rho, self.__S)

        total_p = np.sum(self.__pressure, axis=0)
        q = 0.5 * rho * (np.linalg.norm(self.V) ** 2) * self.__S
        self.__Cp = total_p / q

    def get_info(self):
        print("gamma_magnitude: \n")
        print(self.__gamma_magnitude)
        print("DONE")
        print(f"\nAspect Ratio {self.__AR}")
        print(f"CL_expected {self.__CL_theoretical:.6f} \t CD_ind_expected {self.__CD_theoretical:.6f}")
        print(f"CL_vlm      {self.__CL:.6f}  \t CD_vlm          {self.__CD:.6f}")
        print(f"\n\ntotal_F {str(np.sum(self.__force, axis=0))}")

    def __get_Cxyz(self):
        return get_vlm_Cxyz(self.__force, self.V, self.rho, self.__S)

    @property
    def Cxyz(self):
        return self.__get_Cxyz()
    
    @property
    def CL_theoretical(self):
        return self.__CL_theoretical

    @property
    def CD_theoretical(self):
        return self.__CD_theoretical
    
    @property
    def AR(self):
        return self.__AR

    @property
    def CL(self):
        return self.__CL

    @property
    def CD(self):
        return self.__CD
    @property
    def Cp(self):
        return self.__Cp
    @property
    def force(self):
        return self.__force

    @property
    def pressure(self):
        return self.__pressure





