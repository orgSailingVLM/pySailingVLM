import numpy as np
from numpy.testing import assert_almost_equal


from sailing_vlm.solver.mesher import discrete_segment, make_point_mesh
from sailing_vlm.solver.panels import make_panels_from_le_te_points, make_panels_from_le_points_and_chords

from unittest import TestCase


class TestMesher(TestCase):
    def setUp(self):

        """
               wing span
                  ^
                 y|
                  |
        P3nw------|------P4ne
           |      |      |
           |      |      |
           |      |      |
           |      |      |
           |      +---------------> chord
           |             |      x
           |             |
           |             |
           |             |
        P2sw----root----P1se
        """

        self.c_root = 4.0  # root chord length
        self.c_tip = 2.0  # tip chord length
        self.wing_span = 16  # wing span length

        # Points defining wing

        self.te_se = np.array([self.c_root, 0, 0])
        self.le_sw = np.array([0, 0, 0])

        self.le_nw = np.array([0, self.wing_span, 0])
        self.te_ne = np.array([self.c_tip, self.wing_span, 0])

        # MESH DENSITY
        self.ns = 10  # number of panels spanwise
        self.nc = 5  # number of panels chordwise

    def test_make_discrete_segment(self):
        line = discrete_segment(self.le_sw, self.te_se, self.nc)

        expected_line = np.array(
            [[0., 0., 0.],
             [0.8, 0., 0.],
             [1.6, 0., 0.],
             [2.4, 0., 0.],
             [3.2, 0., 0.],
             [4., 0., 0.]])

        assert np.allclose(line, expected_line)

    def test_make_point_mesh(self):
        s_line = discrete_segment(self.le_sw, self.te_se, self.nc)
        n_line = discrete_segment(self.le_nw, self.te_ne, self.nc)

        mesh = make_point_mesh(s_line, n_line, self.ns)
        expected_mesh0 = np.array(
            [[0., 0., 0.],
             [0., 1.6, 0.],
             [0., 3.2, 0.],
             [0., 4.8, 0.],
             [0., 6.4, 0.],
             [0., 8., 0.],
             [0., 9.6, 0.],
             [0., 11.2, 0.],
             [0., 12.8, 0.],
             [0., 14.4, 0.],
             [0., 16., 0.]])

        assert np.allclose(mesh[0], expected_mesh0)
