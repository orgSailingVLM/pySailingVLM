from unittest import TestCase
import numpy as np
class TimeComparison(TestCase):
    
    def setUp(self):
        self.chord = 1.              # chord length
        self.half_wing_span = 100.
        self.AoA_deg = 3.0
        self.V = 1*np.array([10.0, 0.0, 0.0])
        self.rho = 1.225  # fluid density [kg/m3]

        self.ns = 10    # number of panels (spanwise)
        self.nc = 5     # number of panels (chordwise)
        self.gamma_orientation = 1.0

        
    def test_old_approach(self):
        from sailingVLM.Solver.vlm_solver import calc_circulation
        from sailingVLM.Solver.mesher import make_panels_from_le_te_points
        from sailingVLM.Rotations.geometry_calc import rotation_matrix
        from sailingVLM.Solver.coeff_formulas import get_CL_CD_free_wing
        from sailingVLM.Solver.forces import calc_force_VLM_xyz, calc_pressure
        from sailingVLM.Solver.vlm_solver import is_no_flux_BC_satisfied, calc_induced_velocity

        np.set_printoptions(precision=3, suppress=True)

   
        # Points defining wing (x,y,z) #
        le_NW = np.array([0., self.half_wing_span, 0.])      # leading edge North - West coordinate
        le_SW = np.array([0., -self.half_wing_span, 0.])     # leading edge South - West coordinate

        te_NE = np.array([self.chord, self.half_wing_span, 0.])   # trailing edge North - East coordinate
        te_SE = np.array([self.chord, -self.half_wing_span, 0.])  # trailing edge South - East coordinate

        Ry = rotation_matrix([0, 1, 0], np.deg2rad(self.AoA_deg))
 
        panels, _, _ = make_panels_from_le_te_points(
        [np.dot(Ry, le_SW),
            np.dot(Ry, te_SE),
            np.dot(Ry, le_NW),
            np.dot(Ry, te_NE)],
        [self.nc, self.ns],
        gamma_orientation=self.gamma_orientation)

        rows, cols = panels.shape
        N = rows * cols


        V_app_infw = np.array([self.V for i in range(N)])
  
        ### CALCULATIONS ###
        gamma_magnitude, v_ind_coeff, _ = calc_circulation(V_app_infw, panels)
        V_induced_at_ctrl_p = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
        V_app_fw_at_ctrl_p = V_app_infw + V_induced_at_ctrl_p
        assert is_no_flux_BC_satisfied(V_app_fw_at_ctrl_p, panels)

        Fold, _, _ = calc_force_VLM_xyz(V_app_infw, gamma_magnitude, panels, rho=self.rho)

        F, _, _ = calc_force_VLM_xyz(V_app_infw, gamma_magnitude, panels, self.rho)
        F = F.reshape(N, 3)

        p = calc_pressure(F, panels)

        AR = 2 * self.half_wing_span / self.chord
        S = 2 * self.half_wing_span * self.chord
        CL_expected, CD_ind_expected = get_CL_CD_free_wing(AR, self.AoA_deg)

        total_F = np.sum(F, axis=0)
        q = 0.5 * self.rho * (np.linalg.norm(self.V) ** 2) * S
        CL_vlm = total_F[2] / q
        CD_vlm = total_F[0] / q
        
        print(f"\nAspect Ratio {AR}")
        print(f"CL_expected {CL_expected:.6f} \t CD_ind_expected {CD_ind_expected:.6f}")
        print(f"CL_vlm      {CL_vlm:.6f}  \t CD_vlm          {CD_vlm:.6f}")

        print(f"\n\ntotal_F {str(total_F)}")
        #print(f"total_Fold {str(np.sum(Fold, axis=0))}")
        print("=== END ===")
    
    def test_new_approach(self):
        import numpy as np
        from sailingVLM.NewApproach.vlm import Vlm

        np.set_printoptions(precision=10, suppress=True)


        my_vlm = Vlm(chord=self.chord, half_wing_span=self.half_wing_span, AoA_deg=self.AoA_deg, M=self.ns, N=self.nc, rho=self.rho, gamma_orientation=self.gamma_orientation, V=self.V)

        print("gamma_magnitude: \n")
        print(my_vlm.big_gamma)
        print("DONE")

        ### compare vlm with book formulas ###
        # reference values - to compare with book formulas
        print(f"\nAspect Ratio {my_vlm.AR}")
        print(f"CL_expected {my_vlm.CL_expected:.6f} \t CD_ind_expected {my_vlm.CD_ind_expected:.6f}")
        print(f"CL_vlm      {my_vlm.CL_vlm:.6f}  \t CD_vlm          {my_vlm.CD_vlm:.6f}")
        print(f"\n\ntotal_F {str(np.sum(my_vlm.F, axis=0))}")
        # print(f"total_Fold {str(np.sum(Fold, axis=0))}")
        print("=== END ===")

