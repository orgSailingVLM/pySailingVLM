import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase

from Solver.vortices import \
    v_induced_by_finite_vortex_line, \
    v_induced_by_semi_infinite_vortex_line, \
    v_induced_by_horseshoe_vortex, \
    is_in_vortex_core


class TestVortices(TestCase):
    def test_v_induced_by_finite_vortex_line(self):
        P = np.array([1, 0, 0], dtype=np.float64)
        A = np.array([0, 0, 0], dtype=np.float64)
        B = np.array([0, 1, 0], dtype=np.float64)

        calculated_vel = v_induced_by_finite_vortex_line(P, A, B, gamma=1)
        expected_vel = [0, 0, -0.056269769]

        assert_almost_equal(calculated_vel, expected_vel)

    def test_v_induced_by_semi_infinite_vortex_line(self):
        A = np.array([2, 1, 0], dtype=np.float64)

        p0 = np.array([-1, -3, 0], dtype=np.float64)
        r0 = np.array([-4, 3, 0], dtype=np.float64)

        calculated_vel0 = v_induced_by_semi_infinite_vortex_line(p0, A, r0, gamma=1)
        expected_vel0 = [0, 0, 0.0159154943]
        assert_almost_equal(calculated_vel0, expected_vel0)

        p1 = np.array([5, 5, 0], dtype=np.float64)
        r1 = np.array([4, -3, 0], dtype=np.float64)

        calculated_vel1 = v_induced_by_semi_infinite_vortex_line(p1, A, r1, gamma=1)
        expected_vel1 = [0, 0, 0.0159154943]
        assert_almost_equal(calculated_vel1, expected_vel1)

    def test_v_induced_by_semi_infinite_vortex_line_vs_finite_vortex_line(self):
        A = np.array([123, 456, 789], dtype=np.float64)
        B = np.array([120, 456, 789], dtype=np.float64)

        ctr_point = np.array([12, 34, 56], dtype=np.float64)
        vortex_line_direction = np.array([1, 0, 0], dtype=np.float64)

        calculated_vel_A = v_induced_by_semi_infinite_vortex_line(ctr_point, A, vortex_line_direction, gamma=1)
        calculated_vel_B = v_induced_by_semi_infinite_vortex_line(ctr_point, B, vortex_line_direction, gamma=-1)
        expected_vel = v_induced_by_finite_vortex_line(ctr_point, A, B, gamma=1)

        difference_AB = calculated_vel_A + calculated_vel_B
        assert_almost_equal(difference_AB, expected_vel)

    def test_v_induced_by_horseshoe_vortex(self):
        V = np.array([1, 0, 0], dtype=np.float64)

        ### i,j = 0,1
        ctr_point_01 = np.array([1.5, -5., 0.], dtype=np.float64)
        a_01 = np.array([0.5, 0., 0.], dtype=np.float64)
        b_01 = np.array([0.5, 10., 0.], dtype=np.float64)

        ### i,j = 1,0
        ctr_point_10 = np.array([1.5, 5., 0.], dtype=np.float64)
        a_10 = np.array([0.5, -10., 0.], dtype=np.float64)
        b_10 = np.array([0.5, 0., 0.], dtype=np.float64)

        v01 = v_induced_by_horseshoe_vortex(ctr_point_01, a_01, b_01, V)
        v10 = v_induced_by_horseshoe_vortex(ctr_point_10, a_10, b_10, V)

        assert np.allclose(v01, v10)

    def test_is_in_vortex_core(self):

        # is_in_vortex_core([1., 2., 3.])
        assert not is_in_vortex_core(np.array([[1., 2., 3.]], dtype=np.float64))
        assert is_in_vortex_core(np.array([[1e-10, 1e-10, 1e-10]], dtype=np.float64))

        P = np.array([1e-12, 0, 0], dtype=np.float64)
        A = np.array([0, 0, 0], dtype=np.float64)
        B = np.array([0, 1e-12, 0], dtype=np.float64)

        calculated_vel = v_induced_by_finite_vortex_line(P, A, B, gamma=1)
        assert_almost_equal(calculated_vel, [0, 0, 0])
