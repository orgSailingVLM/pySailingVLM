"""
    vlm_logic tests
"""
from unittest import TestCase
import numpy as np
import numba
from numpy.testing import assert_almost_equal


from sailing_vlm.solver.coefs import calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points, \
    calc_velocity_coefs, solve_eq, get_vlm_CL_CD_free_wing, get_CL_CD_free_wing, calculate_RHS
from sailing_vlm.solver.panels import get_panels_area, make_panels_from_le_te_points
from sailing_vlm.solver.forces import is_no_flux_BC_satisfied 
from sailing_vlm.rotations.geometry_calc import rotation_matrix
from sailing_vlm.solver.velocity import calculate_app_fs 
from sailing_vlm.solver.forces import is_no_flux_BC_satisfied, calc_force_wrapper

class TestCoefs(TestCase):

    def setUp(self):
        
        self.M = 1
        self.N = 1
        self.panels = np.array([[   [  10, 0, 0 ],
                                    [  0, 0, 0  ],
                                    [  0, 10, 0 ],
                                    [  10, 10, 0]]])
        self.gamma_orientation = 1.0
        self.normals = np.array([[0.0, 0.0, -1.0]])
        self.ctr_p = np.array([[7.5, 5.0, 0.0]])
        self.cps = np.array([[2.5, 5.0, 0.0]])
        self.rings = np.array([[[12.5, 0.0, 0.0 ],
                                [2.5, 0.0, 0.0  ],
                                [2.5, 10.0, 0.0 ],
                                [12.5, 10.0, 0.0]]])
        self.spans = np.array([[0.0, 10.0, 0.0]]) * self.gamma_orientation
        self.V = 1 * np.array([10.0, 0.0, 0.0])
        self.rho = 1.225

        self.leading_mid_points = np.array([[0., 5., 0.]])
        self.trailing_mid_points = np.array([[10.,  5.,  0.]])
    def test_calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(self):
        normals, ctr_p, cp, rings, span_vectors, leading_mid_points, trailing_mid_points = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(self.panels, self.gamma_orientation)
    
        assert_almost_equal(normals, self.normals)
        assert_almost_equal(ctr_p, self.ctr_p)
        assert_almost_equal(cp, self.cps)
        assert_almost_equal(rings, self.rings)
        assert_almost_equal(span_vectors, self.spans)
        assert_almost_equal(leading_mid_points, self.leading_mid_points)
        assert_almost_equal(trailing_mid_points, self.trailing_mid_points)


    def test_get_influence_coefficients_spanwise(self):
        panels = np.array([[[  2., -10.        , 0.],
                            [  0., -10.        , 0.],
                            [  0.,  -3.33333333, 0.],
                            [  2.,  -3.33333333, 0.]],

                        [[  2.,  -3.33333333,   0.],
                            [  0.,  -3.33333333,   0.],
                            [  0.,   3.33333333,   0.],
                            [  2.,   3.33333333,   0.]],

                        [[  2., 3.33333333,0.],
                            [  0. , 3.33333333,   0.],
                            [  0.,  10.        ,   0.],
                            [  2.,  10.        ,   0.]]])

        trailing_edge_info = np.array([True, True, True])
        normals, ctr_p, cp, rings, _, _, _ = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels, self.gamma_orientation)
    
        M = 3  # number of panels (spanwise)
        N = 1
        V = 1 * np.array([10.0, 0.0, -1])  # [m/s] wind speed
        V_free_stream = np.array([V for i in range(N * M)])

        coeff, v_ind_coeff = calc_velocity_coefs(V_free_stream, ctr_p, rings, normals, trailing_edge_info, self.gamma_orientation)
        RHS = calculate_RHS(V_free_stream, normals)
        gamma_magnitude = solve_eq(coeff, RHS)
        areas = get_panels_area(panels)
        gamma_expected = [-5.26057, -5.61231, -5.26057]
        assert_almost_equal(gamma_magnitude, gamma_expected, decimal=5)

        V_induced, V_app_fs = calculate_app_fs(V_free_stream, v_ind_coeff, gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs, panels, areas, normals)

        with self.assertRaises(ValueError) as context:
            V_broken = 1e10 * V_app_fs
            is_no_flux_BC_satisfied(V_broken, panels, areas, normals)

        self.assertTrue("Solution error, there shall be no flow through panel!" in context.exception.args[0])

    def test_get_vlm_CL_CD_free_wing(self): 
        gamma_orientation = 1
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
        ns = 5   # number of panels (spanwise)
        nc = 5   # number of panels (chordwise)
        
        panels, trailing_edge_info, leading_edge_info = make_panels_from_le_te_points(
            [np.dot(Ry, le_SW),
            np.dot(Ry, te_SE),
            np.dot(Ry, le_NW),
            np.dot(Ry, te_NE)],
            [nc, ns])

        N = panels.shape[0]

        ### FLIGHT CONDITIONS ###
        V = 1*np.array([10.0, 0.0, 0.0])
        V_app_infw = np.array([V for i in range(N)])
        rho = 1.225  # fluid density [kg/m3]

        AR = 2 * half_wing_span / chord
        S = 2 * half_wing_span * chord
        CL_expected, CD_expected = get_CL_CD_free_wing(AR, AoA_deg)
        
        areas = get_panels_area(panels) 
        normals, ctr_p, cp, rings, span_vectors, _, _ = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels, gamma_orientation)

        coeff, v_ind_coeff = calc_velocity_coefs(V_app_infw, ctr_p, rings, normals, trailing_edge_info, self.gamma_orientation)
        RHS = calculate_RHS(V_app_infw, normals)
        gamma_magnitude = solve_eq(coeff, RHS)

        _,  V_app_fs_at_ctrl_p = calculate_app_fs(V_app_infw,  v_ind_coeff,  gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, panels, areas, normals)

        force, _, _ = calc_force_wrapper(V_app_infw, gamma_magnitude, rho, cp, rings, ns, normals, span_vectors, trailing_edge_info, leading_edge_info, gamma_orientation)

        CL_vlm, CD_vlm = get_vlm_CL_CD_free_wing(force, V, rho, S)
        
        # for small ns and nc this results are okey -> decimal is set in assert functions
        #ACTUAL: 0.32590910827569786
        # DESIRED: 0.32492524777248916
        np.testing.assert_almost_equal(CL_vlm, CL_expected, decimal=3)
        
        # ACTUAL: 0.0001505282134756598
        # DESIRED: 0.00021003760727734467
        np.testing.assert_almost_equal(CD_vlm, CD_expected, decimal=3)

    def test_get_CL_CD_from_coeff(self):
        AR = 20
        AoA_deg = 10
        CL_expected, CD_ind_expected = get_CL_CD_free_wing(AR, AoA_deg)

        assert_almost_equal(CL_expected, 0.974775743317)
        assert_almost_equal(CD_ind_expected, 0.018903384655)
