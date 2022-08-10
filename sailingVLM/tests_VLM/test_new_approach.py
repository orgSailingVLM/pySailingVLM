import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase

from sailingVLM.Solver.vlm_solver import calc_circulation
from sailingVLM.Solver.mesher import make_panels_from_le_te_points
from sailingVLM.Rotations.geometry_calc import rotation_matrix
from sailingVLM.Solver.coeff_formulas import get_CL_CD_free_wing
from sailingVLM.Solver.forces import calc_force_VLM_xyz, calc_pressure
from sailingVLM.Solver.vlm_solver import is_no_flux_BC_satisfied, calc_induced_velocity
from sailingVLM.NewApproach.vlm import Vlm


class TestNewApproach(TestCase):
    
    def test_compare_old_and_new_approach(self):

        np.set_printoptions(precision=10, suppress=True)

        ### WING DEFINITION ###
        #Parameters #
        chord = 1.              # chord length
        half_wing_span = 100.    # wing span length

        # Points defining wing (x,y,z) #
        le_NW = np.array([0., half_wing_span, 0.])      # leading edge North - West coordinate
        le_SW = np.array([0., -half_wing_span, 0.])     # leading edge South - West coordinate

        te_NE = np.array([chord, half_wing_span, 0.])   # trailing edge North - East coordinate
        te_SE = np.array([chord, -half_wing_span, 0.])  # trailing edge South - East coordinate

        AoA_deg = 3.0   # Angle of attack [deg]
        Ry = rotation_matrix([0, 1, 0], np.deg2rad(AoA_deg))
        # we are going to rotate the geometry

        ### MESH DENSITY ###
        ns =  10   # number of panels (spanwise)
        nc = 5   # number of panels (chordwise)

        panels, _, _ = make_panels_from_le_te_points(
            [np.dot(Ry, le_SW),
            np.dot(Ry, te_SE),
            np.dot(Ry, le_NW),
            np.dot(Ry, te_NE)],
            [nc, ns],
            gamma_orientation=1)

        center_of_pressure_good = []
        rings_good = []
        normals_good = []
        for element in panels:
            for panel in element:
                center_of_pressure_good.append(panel.get_cp_position())
                rings_good.append(panel.get_vortex_ring_position())
                normals_good.append(panel.get_normal_to_panel())
                
        center_of_pressure_good = np.array(center_of_pressure_good)
        rings_good = np.array(rings_good) 
        normals_good = np.array(normals_good)

        rows, cols = panels.shape
        N = rows * cols

        ### FLIGHT CONDITIONS ###
        V = 1*np.array([10.0, 0.0, 0.0])
        V_app_infw = np.array([V for i in range(N)])
        rho = 1.225  # fluid density [kg/m3]

        ### CALCULATIONS ###
        gamma_magnitude, v_ind_coeff, A = calc_circulation(V_app_infw, panels)
        V_induced_at_ctrl_p = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
        V_app_fw_at_ctrl_p = V_app_infw + V_induced_at_ctrl_p
        assert is_no_flux_BC_satisfied(V_app_fw_at_ctrl_p, panels)

        # V_app_infw.reshape(ns,nc,3)
        F, _, _ = calc_force_VLM_xyz(V_app_infw, gamma_magnitude, panels, rho)
        F  = F.reshape(N, 3)

        p = calc_pressure(F, panels)


        ### compare vlm with book formulas ###
        # reference values - to compare with book formulas
        AR = 2 * half_wing_span / chord
        S = 2 * half_wing_span * chord
        CL_expected, CD_ind_expected = get_CL_CD_free_wing(AR, AoA_deg)

        total_F = np.sum(F, axis=0)
        q = 0.5 * rho * (np.linalg.norm(V) ** 2) * S
        CL_vlm = total_F[2] / q
        CD_vlm = total_F[0] / q

        ##################### new approach ###########
        my_vlm = Vlm(chord=chord, half_wing_span=half_wing_span, AoA_deg=AoA_deg, M=cols, N=rows, rho=rho, gamma_orientation=1.0, V=V)

        assert_almost_equal(center_of_pressure_good, my_vlm.center_of_pressure)
        assert_almost_equal(rings_good, my_vlm.rings)
        assert_almost_equal(gamma_magnitude, my_vlm.big_gamma)
        assert_almost_equal(v_ind_coeff, my_vlm.wind_coefs)
        assert_almost_equal(normals_good, my_vlm.normals)
        assert_almost_equal(F, my_vlm.F)

        assert_almost_equal(p, my_vlm.pressure)
        assert_almost_equal(CL_vlm, my_vlm.CL_vlm)
        assert_almost_equal(CD_vlm, my_vlm.CD_vlm)

        assert_almost_equal(CL_expected, my_vlm.CL_expected)
        assert_almost_equal(CD_ind_expected, my_vlm.CD_ind_expected)