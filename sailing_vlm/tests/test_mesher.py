import numpy as np
from numpy.testing import assert_almost_equal

from sailingVLM.Solver.Panel import Panel
from sailingVLM.Solver.mesher import discrete_segment, make_point_mesh
from sailingVLM.Solver.mesher import make_panels_from_le_te_points, make_panels_from_le_points_and_chords

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

    def test_make_panels_from_points_span_and_chord_wise(self):
        panels_le_te, mesh_le_te, new_approach_panels = make_panels_from_le_te_points(
            [self.le_sw, self.te_se,
             self.le_nw, self.te_ne],
            [self.nc, self.ns],
            gamma_orientation=1)

        chords = np.linspace(self.c_root, self.c_tip, num=self.ns+1, endpoint=True)
        chords_vec = np.array([chords, np.zeros(len(chords)), np.zeros(len(chords))])
        chords_vec = chords_vec.transpose()
        panels_c, mesh_c = make_panels_from_le_points_and_chords(
            [self.le_sw, self.le_nw],
            [self.nc, self.ns],
            chords_vec,
            gamma_orientation=-1)

        expected_points = (np.array([0.8, 0., 0.]), np.array([0., 0., 0.]), np.array([0., 1.6, 0.]), np.array([0.76, 1.6, 0.]))
        assert np.allclose(expected_points, panels_c[0, 0].get_points())
        assert np.allclose(expected_points, panels_le_te[0, 0].get_points())

        expected_points37 = (np.array([2.08, 11.2, 0.]), np.array([1.56, 11.2,  0.]), np.array([1.44, 12.8, 0.]), np.array([1.92, 12.8,  0.]))
        assert np.allclose(expected_points37, panels_le_te[3, 7].get_points())
        assert np.allclose(expected_points37, panels_c[3, 7].get_points())

    def test_make_panels_from_points(self):
        panels, _, _ = make_panels_from_le_te_points(
            [self.le_sw, self.te_se,
             self.le_nw, self.te_ne],
            [self.nc, self.ns],
            gamma_orientation=1)

        expected_panel = Panel(np.array([0.68, 4.8, 0.]),
                               np.array([0., 4.8, 0.]),
                               np.array([0., 6.4, 0.]),
                               np.array([0.64, 6.4, 0.]))

        assert np.allclose(panels[0][3].p1, expected_panel.p1)
        assert np.allclose(panels[0][3].p2, expected_panel.p2)
        assert np.allclose(panels[0][3].p3, expected_panel.p3)
        assert np.allclose(panels[0][3].p4, expected_panel.p4)

    def test_dimensions(self):
        panels, _, _ = make_panels_from_le_te_points(
            [self.le_sw, self.te_se,
             self.le_nw, self.te_ne],
            [self.nc, self.ns],
            gamma_orientation=1)

        rows, cols = panels.shape

        assert cols == self.ns
        assert rows == self.nc
