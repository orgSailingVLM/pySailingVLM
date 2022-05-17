import numpy as np
from numpy.testing import assert_almost_equal

from sailingVLM.NewApproach import vlm
from unittest import TestCase


class TestPanels(TestCase):
    def setUp(self):
        
        self.points = np.array([[[10., 0.,   0.],
            [ 0., 0., 0.],
            [ 0., 10., 0.],
            [  10., 10., 0.]],
        ])
        self.N = 1
        self.M = 1
        
        #self.panel = Panel(*self.points)

        #self.assertTrue(self.panel._are_points_coplanar())



    def test_area(self):

        calculated_area = vlm.Panels.get_panels_area(self.points, self.N, self.M)
        expected_area = 100.0

        assert_almost_equal(calculated_area, expected_area)
