"""
    Unit tests_LLT_optimizer of the Panel class and its methods

"""

import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase


from sailing_vlm.solver.coefs import calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points, \
                                            get_influence_coefficients_spanwise

from sailing_vlm.solver.panels import get_panels_area, make_panels_from_le_te_points

class TestTrailingEdgePanel(TestCase):
    def setUp(self):
        self.points = [np.array([0., 0., 0.]), np.array([10., 0., 0.]),
                    np.array([0., 10., 0.]), np.array([10., 10., 0.])]

       
        self.gamma_orientation = 1
        self.trailing_edge_info = np.array([True])
    def test_get_horse_shoe_induced_velocity(self):
        panels, _, _ = make_panels_from_le_te_points(self.points, [1, 1])
        dummy_velocity = np.array([[0.1, 0.2, 0.3]])

        areas = get_panels_area(panels) 
        normals, collocation_points, _, rings, _, _, _ = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels, self.gamma_orientation)

        coefs, RHS, wind_coefs = get_influence_coefficients_spanwise(collocation_points, rings, normals, dummy_velocity, self.trailing_edge_info, self.gamma_orientation)
        v_ind_expected = np.array([[0.0200998, -0.0093672, -0.022963]])

        assert_almost_equal(wind_coefs[0], v_ind_expected)
   