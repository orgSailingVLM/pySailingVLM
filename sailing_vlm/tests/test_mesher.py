import numpy as np
from unittest import TestCase

from sailing_vlm.solver.mesher import discrete_segment, make_point_mesh, make_airfoil_mesh
from sailing_vlm.solver.interpolator import Interpolator
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

    def test_make_airfoil_mesh(self):
        le_sw = np.array([0., 0., 0.])
        le_nw = np.array([0., 0., self.wing_span])
        girths = np.array([0., 1./4, 1./2, 3./4, 1.])
        chords = np.array([5.0]*len(girths))
        
        distance_from_luff = np.array([0.1]*len(girths)) # with camer =0 it do nothing but you cannot set it to zero!
        camber = 0*np.array([0.1]*len(girths))
        interpolator = Interpolator('linear')
        interpolated_chords=interpolator.interpolate_girths(girths, chords, self.ns + 1)
        interpolated_camber=interpolator.interpolate_girths(girths, camber, self.ns + 1)
        interpolated_distance_from_luff=interpolator.interpolate_girths(girths, distance_from_luff, self.ns + 1)

        chords_vec = np.array([interpolated_chords, np.zeros(len(interpolated_chords)), np.zeros(len(interpolated_chords))])
        chords_vec = chords_vec.transpose()
        #interpolator
        mesh = make_airfoil_mesh([le_sw, le_nw], [self.nc, self.ns], chords_vec, interpolated_distance_from_luff, interpolated_camber)

        
        meesh_good_0 = np.array([[0., 0., 0.],
                                [1., 0., 0.],
                                [2., 0., 0.],
                                [3., 0., 0.],
                                [4., 0., 0.],
                                [5., 0., 0.]])


        mesh_good_last = np.array([[ 0.,  0., 16.],
                                    [ 1.,  0., 16.],
                                    [ 2.,  0., 16.],
                                    [ 3.,  0., 16.],
                                    [ 4.,  0., 16.],
                                    [ 5.,  0., 16.]])

        mesh_good_2 = np.array([[0. , 0. , 3.2],
                                [1. , 0. , 3.2],
                                [2. , 0. , 3.2],
                                [3. , 0. , 3.2],
                                [4. , 0. , 3.2],
                                [5. , 0. , 3.2]])
        
        np.testing.assert_almost_equal(mesh[0], meesh_good_0)
        np.testing.assert_almost_equal(mesh[2], mesh_good_2)
        np.testing.assert_almost_equal(mesh[-1], mesh_good_last)