"""
    vlm_logic tests
"""
from unittest import TestCase
import numpy as np
import numba
import os
from numpy.testing import assert_almost_equal


from sailingVLM.NewApproach.vlm_logic import calculate_normals_collocations_cps_rings_spans, \
                                            vortex_line, \
                                            vortex_infinite_line, \
                                            vortex_horseshoe, \
                                            is_in_vortex_core, \
                                            vortex_ring, \
                                            get_influence_coefficients_spanwise, \
                                            solve_eq, \
                                            calc_induced_velocity, \
                                            get_panels_area, \
                                            is_no_flux_BC_satisfied, \
                                            calc_force_wrapper_new, \
                                            calc_pressure, \
                                            get_vlm_CL_CD_free_wing, \
                                            make_panels_from_le_te_points_new, \
                                            create_panels
from sailingVLM.Solver.coeff_formulas import get_CL_CD_free_wing


class TestVlmLogic(TestCase):

    def setUp(self):
        
        self.M = 1
        self.N = 1
        self.panels = np.array([[   [  10, 0, 0 ],
                                    [  0, 0, 0  ],
                                    [  0, 10, 0 ],
                                    [  10, 10, 0]]])
        self.gamma_orientation = 1.0
        self.normals = np.array([[0.0, 0.0, 1.0]])
        self.collocations = np.array([[7.5, 5.0, 0.0]])
        self.cps = np.array([[2.5, 5.0, 0.0]])
        self.rings = np.array([[[12.5, 0.0, 0.0 ],
                                [2.5, 0.0, 0.0  ],
                                [2.5, 10.0, 0.0 ],
                                [12.5, 10.0, 0.0]]])
        self.spans = np.array([[0.0, 10.0, 0.0]]) * self.gamma_orientation
        self.V = 1 * np.array([10.0, 0.0, 0.0])
        self.rho = 1.225
     
    def test_calculate_normals_collocations_cps_rings_spans(self):
        normals, collocations, cps, rings, spans = calculate_normals_collocations_cps_rings_spans(self.panels, self.gamma_orientation)

        assert_almost_equal(normals, self.normals)
        assert_almost_equal(collocations, self.collocations)
        assert_almost_equal(cps, self.cps)
        assert_almost_equal(rings, self.rings)
        assert_almost_equal(spans, self.spans)

    def test_vortex_line(self):
        p1 = self.panels[0][0]
        p2 = self.panels[0][1]
        p = np.array([1.0, 1.0, 1.0])

        q = vortex_line(p, p1, p2)
        q2 = vortex_line(p, p1, p1)

        q_good = np.array([ 0.0,  0.0622785, -0.0622785])
        q_good2 = np.array([ 0.0, 0.0, 0.0])

        assert_almost_equal(q, q_good)
        assert_almost_equal(q2, q_good2)

    def test_vortex_infinite_line(self):
        A = np.array([2, 1, 0])

        p0 = np.array([-1, -3, 0])
        r0 = np.array([-4, 3, 0])

        calculated_vel0 = vortex_infinite_line(p0, A, r0)
        expected_vel0 = [0, 0, 0.0159154943]
        assert_almost_equal(calculated_vel0, expected_vel0)

        p1 = np.array([5, 5, 0])
        r1 = np.array([4, -3, 0])

        calculated_vel1 = vortex_infinite_line(p1, A, r1)
        expected_vel1 = [0, 0, 0.0159154943]
        assert_almost_equal(calculated_vel1, expected_vel1)

    def test_vortex_horseshoe(self):
        V = [1, 0, 0]

        ### i,j = 0,1
        ctr_point_01 = np.array([1.5, -5., 0.])
        a_01 = np.array([0.5, 0., 0.])
        b_01 = np.array([0.5, 10., 0.])

        ### i,j = 1,0
        ctr_point_10 = np.array([1.5, 5., 0.])
        a_10 = np.array([0.5, -10., 0.])
        b_10 = np.array([0.5, 0., 0.])

        v01 = vortex_horseshoe(ctr_point_01, a_01, b_01, V)
        v10 = vortex_horseshoe(ctr_point_10, a_10, b_10, V)

        assert np.allclose(v01, v10)

    def test_is_in_vortex_core(self):
        assert not is_in_vortex_core(numba.typed.List([1, 2, 3]))
        assert is_in_vortex_core(numba.typed.List([1e-10, 1e-10, 1e-10]))

        P = np.array([1e-12, 0, 0])
        A = np.array([0, 0, 0])
        B = np.array([0, 1e-12, 0])

        calculated_vel = vortex_line(P, A, B)
        assert_almost_equal(calculated_vel, [0, 0, 0])

    def test_vortex_ring(self):
        ring = self.rings[0]
        v_ind = vortex_ring(self.collocations[0], ring[0], ring[1], ring[2], ring[3])
        v_ind_expected = [0, 0, -0.09003163161571061]

        assert_almost_equal(v_ind, v_ind_expected)
    def test_get_influence_and_magnitude_coefficients_spanwise(self):
        # wg testow z wersji pierwotnej kodu
        panels = np.array([[[  2.        , -10.        ,   0.        ],
                            [  0.        , -10.        ,   0.        ],
                            [  0.        ,  -3.33333333,   0.        ],
                            [  2.        ,  -3.33333333,   0.        ]],

                        [[  2.        ,  -3.33333333,   0.        ],
                            [  0.        ,  -3.33333333,   0.        ],
                            [  0.        ,   3.33333333,   0.        ],
                            [  2.        ,   3.33333333,   0.        ]],

                        [[  2.        ,   3.33333333,   0.        ],
                            [  0.        ,   3.33333333,   0.        ],
                            [  0.        ,  10.        ,   0.        ],
                            [  2.        ,  10.        ,   0.        ]]])

        normals, collocations, _, rings, _ = calculate_normals_collocations_cps_rings_spans(panels, self.gamma_orientation)

        M = 3  # number of panels (spanwise)
        N = 1
        V = 1 * np.array([10.0, 0.0, -1])  # [m/s] wind speed
        V_free_stream = np.array([V for i in range(N * M)])

        coefs, RHS, wind_coefs = get_influence_coefficients_spanwise(collocations, rings, normals, M, N, V_free_stream )
        gamma_magnitude = solve_eq(coefs, RHS)
        areas = get_panels_area(panels, N, M)

        gamma_expected = [-5.26437093, -5.61425005, -5.26437093]
        assert_almost_equal(gamma_magnitude, gamma_expected)

        V_induced = calc_induced_velocity(wind_coefs, gamma_magnitude)
        V_app_fs = V_free_stream + V_induced
        assert is_no_flux_BC_satisfied(V_app_fs, panels, areas, normals)

        with self.assertRaises(ValueError) as context:
            V_broken = 1e10 * V_app_fs
            is_no_flux_BC_satisfied(V_broken, panels, areas, normals)

        self.assertTrue("Solution error, there shall be no flow through panel!" in context.exception.args[0])

    def test_make_panels_from_points_span(self):
        c_root = 4.0  # root chord length
        c_tip = 2.0  # tip chord length
        wing_span = 16  # wing span length

        # Points defining wing

        te_se = np.array([c_root, 0, 0])
        le_sw = np.array([0, 0, 0])

        le_nw = np.array([0, wing_span, 0])
        te_ne = np.array([c_tip, wing_span, 0])

        # MESH DENSITY
        ns = 10  # number of panels spanwise
        nc = 5  # number of panels chordwise


        new_approach_panels = make_panels_from_le_te_points_new(
            [le_sw, te_se,
            le_nw, te_ne],
            [nc, ns],
            gamma_orientation=1)

        chords = np.linspace(c_root, c_tip, num=ns+1, endpoint=True)
        chords_vec = np.array([chords, np.zeros(len(chords)), np.zeros(len(chords))])
        chords_vec = chords_vec.transpose()

        expected_points = (np.array([0.8, 0., 0.]), np.array([0., 0., 0.]), np.array([0., 1.6, 0.]), np.array([0.76, 1.6, 0.]))

        assert np.allclose(expected_points, new_approach_panels[0])

        expected_points37 = (np.array([2.08, 11.2, 0.]), np.array([1.56, 11.2,  0.]), np.array([1.44, 12.8, 0.]), np.array([1.92, 12.8,  0.]))
        assert np.allclose(expected_points37, new_approach_panels[37])

    def test_get_vlm_CL_CD_free_wing(self):
        ### MESH DENSITY ###
        M = 20  # number of panels (spanwise)
        N = 1   # number of panels (chordwise)
        chord  = 1.              # chord length
        half_wing_span = 100
        AoA_deg = 3.0   # Angle of attack [deg]

        panels = create_panels(half_wing_span, chord, AoA_deg, M, N)
        normals, collocations, cps, rings, spans = calculate_normals_collocations_cps_rings_spans(panels, self.gamma_orientation)
        V_app_infw = np.array([self.V for i in range(M * N)])
        coefs, RHS, wind_coefs = get_influence_coefficients_spanwise(collocations, rings, normals, M, N, V_app_infw)
        gamma_magnitude = solve_eq(coefs, RHS)
        areas = get_panels_area(panels, N, M)

        V_induced = calc_induced_velocity(wind_coefs, gamma_magnitude)
        V_app_fw = V_app_infw + V_induced

        assert is_no_flux_BC_satisfied(V_app_fw, panels, areas, normals)

        F = calc_force_wrapper_new(V_app_infw, gamma_magnitude, panels, self.rho, cps, rings, M, N, normals, spans)
        S = 2 * half_wing_span * chord
        AR = 2 * half_wing_span / chord

        CL_expected, CD_exptected = get_CL_CD_free_wing(AR, AoA_deg)

        CL_vlm, CD_vlm = get_vlm_CL_CD_free_wing(F, self.V, self.rho, S)

        assert_almost_equal(CL_vlm, CL_expected, decimal=3)
        assert_almost_equal(CD_vlm, CD_exptected, decimal=3)
        