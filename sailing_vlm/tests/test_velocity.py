"""
    vlm_logic tests
"""
from unittest import TestCase
import numpy as np
import numba
import os
from numpy.testing import assert_almost_equal

from sailing_vlm.solver.velocity import vortex_line, vortex_infinite_line, \
                                    vortex_horseshoe, vortex_ring, \
                                    is_in_vortex_core, calc_induced_velocity
from sailing_vlm.solver.panels import get_panels_area
from sailing_vlm.solver.coefs import get_influence_coefficients_spanwise, solve_eq

class TestVelocity(TestCase):

    def setUp(self):
        
        self.nc = 1
        self.ns = 1
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

        self.leading_mid_points = np.array([[0., 5., 0.]])
        self.trailing_mid_points = np.array([[10.,  5.,  0.]])
        self.trailing_edge_info = np.array([True])

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

    def test_calc_induced_velocity(self):
        V = 1 * np.array([10.0, 0.0, -1])  # [m/s] wind speed
        V_free_stream = np.array([V for i in range(self.ns * self.nc)])

        coefs, RHS, wind_coefs = get_influence_coefficients_spanwise(self.collocations, self.rings, self.normals, V_free_stream, self.trailing_edge_info, self.gamma_orientation)
        gamma_magnitude = solve_eq(coefs, RHS)
        areas = get_panels_area(self.panels)

        gamma_expected = [-13.168814113460344]
        np.testing.assert_almost_equal(gamma_magnitude, gamma_expected)

        V_induced_expected = np.array([[0.07035975, 0.        , 1.        ]])
        V_induced = calc_induced_velocity(wind_coefs, gamma_magnitude)
        np.testing.assert_almost_equal(V_induced, V_induced_expected)
        
 
    
    def test_v_induced_by_vortex_line_vs_vortex_infinite_line(self):
        A = np.array([123, 456, 789], dtype=np.float64)
        B = np.array([120, 456, 789], dtype=np.float64)

        ctr_point = np.array([12, 34, 56], dtype=np.float64)
        vortex_line_direction = np.array([1, 0, 0], dtype=np.float64)
        # calculated_vel_A ma zal wartosc, patzr pomozej
        calculated_vel_A = vortex_line(ctr_point, A, vortex_line_direction, gamma=1)
        calculated_vel_B = vortex_infinite_line(ctr_point, B, vortex_line_direction, gamma=-1)
        expected_vel = vortex_line(ctr_point, A, B, gamma=1)

        difference_AB = calculated_vel_A + calculated_vel_B
        assert_almost_equal(difference_AB, expected_vel)
        
# array([ 0.00000000e+00,  7.09284116e-05, -4.08346381e-05])
# calculated_vel_B
# array([-0.00000000e+00, -7.12105017e-05,  4.09970419e-05])
# expected_vel
# array([ 0.00000000e+00, -2.82090017e-07,  1.62403802e-07])
 
    def test_v_induced_by_finite_vortex_line(self):
        P = np.array([1, 0, 0], dtype=np.float64)
        A = np.array([0, 0, 0], dtype=np.float64)
        B = np.array([0, 1, 0], dtype=np.float64)

        calculated_vel = vortex_line(P, A, B, gamma=1)
        expected_vel = [0, 0, -0.056269769]

        assert_almost_equal(calculated_vel, expected_vel)
