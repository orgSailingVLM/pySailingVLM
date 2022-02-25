"""
    Unit tests_LLT_optimizer of the Panel class and its methods

"""

import numpy as np
from numpy.testing import assert_almost_equal

from Solver.TrailingEdgePanel import TrailingEdgePanel
from unittest import TestCase


class TestTrailingEdgePanel(TestCase):
    def setUp(self):
        self.points = [np.array([10, 0, 0]), np.array([0, 0, 0]),
                       np.array([0, 10, 0]), np.array([10, 10, 0])]

        self.panel = TrailingEdgePanel(*self.points)
        self.assertTrue(self.panel._are_points_coplanar())

    def test_get_horse_shoe_induced_velocity(self):
        ctr_p = self.panel.get_ctr_point_position()
        dummy_velocity = np.array([0.1, 0.2, 0.3])
        v_ind = self.panel.get_induced_velocity(ctr_p, dummy_velocity)
        v_ind_expected = [0.0200998, -0.0093672, -0.022963]

        assert_almost_equal(v_ind, v_ind_expected)

