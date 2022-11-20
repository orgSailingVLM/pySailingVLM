import numpy as np
from unittest import TestCase

from sailing_vlm.solver.panels import make_panels_from_le_te_points, make_panels_from_le_points_and_chords, get_panels_area


class TestPanels(TestCase):
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

    def test_make_panels_from_points_span_and_chord_wise(self):
        panels_le_te, trailing_edge_info_le_te, leading_edge_info_le_te  = make_panels_from_le_te_points(
            [self.le_sw, self.te_se,
             self.le_nw, self.te_ne],
            [self.nc, self.ns])

        chords = np.linspace(self.c_root, self.c_tip, num=self.ns+1, endpoint=True)
        chords_vec = np.array([chords, np.zeros(len(chords)), np.zeros(len(chords))])
        chords_vec = chords_vec.transpose()
        panels_c, trailing_edge_info_c, leading_edge_info_c = make_panels_from_le_points_and_chords(
            [self.le_sw, self.le_nw],
            [self.nc, self.ns],
            chords_vec,
            gamma_orientation=-1)

        expected_points = (np.array([0.8, 0., 0.]), np.array([0., 0., 0.]), np.array([0., 1.6, 0.]), np.array([0.76, 1.6, 0.]))
        assert np.allclose(expected_points, panels_c[0])
        assert np.allclose(expected_points, panels_le_te[0])

        expected_points37 = (np.array([2.08, 11.2, 0.]), np.array([1.56, 11.2,  0.]), np.array([1.44, 12.8, 0.]), np.array([1.92, 12.8,  0.]))
        assert np.allclose(expected_points37, panels_le_te[37])
        assert np.allclose(expected_points37, panels_c[37])
        
        # check trailing arrays
        start = trailing_edge_info_le_te.shape[0] - self.ns
        assert np.all(trailing_edge_info_le_te[start:-1])
        assert np.all(trailing_edge_info_c[start:-1])

        # test leading edges array
        assert np.all(leading_edge_info_le_te[0:self.ns])
        assert np.all(leading_edge_info_c[0:self.ns])
    
    def test_get_panels_area(self):
        points = [np.array([10., 0., 0.]), np.array([0., 0., 0.]),
                    np.array([0., 10., 0.]), np.array([10., 10., 0.])]

        panels, _, _ = make_panels_from_le_te_points(points, [1, 1])
        calculated_area = get_panels_area(panels)
        expected_area = 100.0

        np.testing.assert_almost_equal(calculated_area, expected_area)
    
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

        new_approach_panels, _, _ = make_panels_from_le_te_points(
            [le_sw, te_se,
            le_nw, te_ne],
            [nc, ns])

        chords = np.linspace(c_root, c_tip, num=ns+1, endpoint=True)
        chords_vec = np.array([chords, np.zeros(len(chords)), np.zeros(len(chords))])
        chords_vec = chords_vec.transpose()

        expected_points = (np.array([0.8, 0., 0.]), np.array([0., 0., 0.]), np.array([0., 1.6, 0.]), np.array([0.76, 1.6, 0.]))

        assert np.allclose(expected_points, new_approach_panels[0])

        expected_points37 = (np.array([2.08, 11.2, 0.]), np.array([1.56, 11.2,  0.]), np.array([1.44, 12.8, 0.]), np.array([1.92, 12.8,  0.]))
        assert np.allclose(expected_points37, new_approach_panels[37])