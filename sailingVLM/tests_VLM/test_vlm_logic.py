from numpy.testing import assert_almost_equal
from unittest import TestCase
import numpy as np
from sailingVLM.NewApproach.vlm_logic import calculate_normals_collocations_cps_rings_spans, \
                                            vortex_line, \
                                            vortex_infinite_line, \
                                            vortex_horseshoe, \
                                            is_in_vortex_core, \
                                            vortex_ring, \
                                            get_influence_coefficients, \
                                            get_influence_coefficients_spanwise, \
                                            solve_eq, \
                                            calc_induced_velocity, \
                                            get_panels_area, \
                                            is_no_flux_BC_satisfied, \
                                            calc_V_at_cp_new, \
                                            calc_force_wrapper_new, \
                                            calc_pressure, \
                                            get_vlm_CL_CD_free_wing, \
                                            make_panels_from_mesh_spanwise_new, \
                                            make_panels_from_le_te_points_new, \
                                            create_panels


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
        assert not is_in_vortex_core([1, 2, 3])
        assert is_in_vortex_core([1e-10, 1e-10, 1e-10])

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
    
    """
    def test_get_influence_coefficients_spanwise(self):
        
        coefs, RHS, wind_coefs, trailing_rings = get_influence_coefficients_spanwise(self.collocations, self.rings, self.normals, self.M, self.N, self.V )
    """