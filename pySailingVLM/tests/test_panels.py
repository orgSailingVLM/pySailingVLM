import numpy as np
from unittest import TestCase

from pySailingVLM.solver.panels import make_panels_from_le_te_points, get_panels_area
from pySailingVLM.yacht_geometry.sail_geometry import SailGeometry
from pySailingVLM.solver.interpolator import Interpolator
from pySailingVLM.rotations.csys_transformations import CSYS_transformations
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

    def test_flat_plate_panels(self):
        # 10 spanwise and 5 chordwise
        # chord length = 5
        # span length = 10 for better calculations
        # FLAT PLATE without camber! 
        
        # (0, 0, 10)--- ... ---(10, 0, 10)
        # |
        # .
        # .
        # .
        # |
        # |
        # (0, 0, 0) --- ... --(0, 0, 10)
        head_mounting = np.array([0., 0., 10])
        tack_mounting = np.array([0., 0., 0])
        heel_deg = 0.
        leeway_deg = 0.
        interpolator = Interpolator('linear')
        reference_level_for_moments = np.array([0, 0, 0])
        girths = np.array([0., 1./4, 1./2, 3./4, 1.])
        chords = np.array([5.00]* len(girths))
        chords_interp = interpolator.interpolate_girths(girths, chords, self.ns + 1)
        camber= 0*np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        camber_distance_from_luff = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) # starting from leading edge
        centerline_twist_deg = 0*girths
        twist_interp = interpolator.interpolate_girths(girths, centerline_twist_deg, self.ns + 1)
      
        interpolated_camber=interpolator.interpolate_girths(girths, camber, self.ns + 1)
        interpolated_distance_from_luff=interpolator.interpolate_girths(girths, camber_distance_from_luff, self.ns + 1)

        
        csys_transformations = CSYS_transformations(
        heel_deg, leeway_deg, v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

        flat_geom = SailGeometry(head_mounting, tack_mounting, csys_transformations, self.ns, self.nc, chords_interp, twist_interp, 'main', 'real_twist', interpolated_camber, interpolated_distance_from_luff)
        
        
        expected_9 = np.array([[ 1.,  0.,  9.],
                [ 0.,  0.,  9.],
                [ 0.,  0., 10.],
                [ 1.,  0., 10.]])
    
        expected_50 = np.array([[  1.,   0., -10.],
                                [  0.,   0., -10.],
                                [  0.,   0.,  -9.],
                                [  1.,   0.,  -9.]])

        expected_15 = np.array([[2., 0., 5.],
                                [1., 0., 5.],
                                [1., 0., 6.],
                                [2., 0., 6.]])
        np.testing.assert_almost_equal( flat_geom.panels[9], expected_9)
        np.testing.assert_almost_equal( flat_geom.panels[50], expected_50)
        np.testing.assert_almost_equal( flat_geom.panels[15], expected_15)
        assert np.all(flat_geom.trailing_edge_info[-10:])
        assert np.all(flat_geom.leading_edge_info[:9])    
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