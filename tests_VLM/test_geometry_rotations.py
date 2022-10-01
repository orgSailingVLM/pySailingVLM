import numpy as np
from numpy.testing import assert_almost_equal

from Rotations.geometry_calc import rotation_matrix
from unittest import TestCase
from YachtGeometry.SailFactory import SailFactory
from YachtGeometry.SailGeometry import SailSet
from Solver.Interpolator import Interpolator

# from sailingVLM.forces import from_xyz_to_centerline_csys
from Rotations.CSYS_transformations import CSYS_transformations
from Solver.PanelsPlotter import _prepare_geometry_data_to_display


class TestRotations(TestCase):
    def setUp(self):
        self.interpolator = Interpolator("linear")
        leeway_deg = 10
        heel_deg = 30
        self.csys_transformations = CSYS_transformations(heel_deg, leeway_deg)

    def test_from_xyx_to_centerline_csys(self):
        #  https://en.wikipedia.org/wiki/Rotation_of_axes
        #  https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3&ab_channel=3Blue1Brown
        leeway_deg = 3.5
        F_xyz = np.array([-3.09, 197.53, -93.1])

        M = np.array([
            [np.cos(np.deg2rad(leeway_deg)), -np.sin(np.deg2rad(leeway_deg)), 0],
            [np.sin(np.deg2rad(leeway_deg)), np.cos(np.deg2rad(leeway_deg)), 0],
            [0., 0., 1.]
        ])

        expected_F_centerline = np.dot(M, F_xyz)
        csys_transformations = CSYS_transformations(heel_deg=0, leeway_deg=leeway_deg)
        F_centerline = csys_transformations.from_xyz_to_centerline_csys([F_xyz])[0]

        assert_almost_equal(expected_F_centerline, F_centerline)

    def test_rotation_matrix(self):
        Ry = rotation_matrix([0, 1, 0], np.deg2rad(45))
        result = np.dot(Ry, [1, 456, 1])

        expected_result = [1.41421356, 456, 0]

        assert_almost_equal(expected_result, result)

    def test_csys_rotations(self):
        point = np.array([10, 20, 30])

        rotated_point = self.csys_transformations.rotate_point_with_mirror(point)
        point_back = self.csys_transformations.reverse_rotations_with_mirror(rotated_point)
        assert_almost_equal(point, point_back)
        assert_almost_equal(np.array([4.235680201509421, 33.56596871090129, 15.980762113533162]), rotated_point)

    def test_sail_set_reverse_rotations(self):
        n_spanwise = 10  # No of control points (above the water) per sail

        main_sail_girths = np.array([0.00, 1. / 4, 1. / 2, 3. / 4, 7. / 8, 1.00])
        main_sail_chords = np.array([4.00, 3.64, 3.20, 2.64, 2.32, 2.00])
        main_sail_luff = 12.4
        sheer_above_waterline = 1.20
        boom_above_sheer = 1.30
        rake_deg = 110  # rake angle [deg]

        sail_factory = SailFactory(self.csys_transformations, n_spanwise=n_spanwise,
                                   rake_deg=rake_deg,
                                   sheer_above_waterline=sheer_above_waterline)

        main_sail_geometry = sail_factory.make_main_sail(main_sail_luff=main_sail_luff,
                                                         boom_above_sheer=boom_above_sheer,
                                                         main_sail_chords=self.interpolator.interpolate_girths(
                                                             main_sail_girths,
                                                             main_sail_chords,
                                                             n_spanwise + 1))

        point = np.array([10, 20, 30])
        rotated_point = main_sail_geometry.csys_transformations.rotate_point_with_mirror(point)
        point_back = main_sail_geometry.csys_transformations.reverse_rotations_with_mirror(rotated_point)
        assert_almost_equal(point, point_back)

    def _prepare_sail_set(self, initial_twist_factor, n_spanwise=10, n_chordwise=1):
        # n_spanwise # No of control points (above the water) per sail

        main_sail_girths = np.array([0.00, 1. / 4, 1. / 2, 3. / 4, 7. / 8, 1.00])
        main_sail_chords = np.array([4.00, 3.64, 3.20, 2.64, 2.32, 2.00])
        main_sail_centerline_twist_deg = initial_twist_factor * main_sail_girths + 0.

        jib_girths = np.array([0.00, 1. / 4, 1. / 2, 3. / 4, 1.00])
        jib_chords = np.array([3.80, 2.98, 2.15, 1.33, 0.50])
        jib_centerline_twist_deg = initial_twist_factor * jib_girths + 0.  #

        main_sail_luff = 12.4
        jib_luff = 10.0
        foretriangle_height = 11.50
        foretriangle_base = 3.90
        sheer_above_waterline = 1.20
        boom_above_sheer = 1.30
        rake_deg = 110  # rake angle [deg]
        LLT_twist = "real_twist"  # defines how the Lifting Line discretize the sail twist.
        # It can be "sheeting_angle_const" or "average_const" or "real_twist"
        sail_factory = SailFactory(self.csys_transformations, n_spanwise=n_spanwise, n_chordwise=n_chordwise,
                                   rake_deg=rake_deg,
                                   sheer_above_waterline=sheer_above_waterline)

        main_sail_geometry = sail_factory.make_main_sail(
            main_sail_luff=main_sail_luff,
            boom_above_sheer=boom_above_sheer,
            main_sail_chords=self.interpolator.interpolate_girths(main_sail_girths, main_sail_chords, n_spanwise + 1),
            sail_twist_deg=self.interpolator.interpolate_girths(main_sail_girths, main_sail_centerline_twist_deg,n_spanwise + 1),
            LLT_twist=LLT_twist)

        jib_geometry = sail_factory.make_jib(
            jib_luff=jib_luff,
            foretriangle_base=foretriangle_base,
            foretriangle_height=foretriangle_height,
            jib_chords=self.interpolator.interpolate_girths(jib_girths, jib_chords, n_spanwise + 1),
            sail_twist_deg=self.interpolator.interpolate_girths(jib_girths, jib_centerline_twist_deg, n_spanwise + 1),
            LLT_twist=LLT_twist)

        sail_set = SailSet([jib_geometry, main_sail_geometry])
        # jib_geometry.display_geometry()
        # sail_set.display_geometry_xz()
        # display_panels_xyz(sail_set.panels1d)

        return sail_set

    def test_data_extractor(self):
        from Solver.forces import extract_above_water_quantities
        sail_set = self._prepare_sail_set(initial_twist_factor=10., n_spanwise=5, n_chordwise=3)
        cp_points = sail_set.get_cp_points1d()

        # assert_almost_equal(cp_points[30], np.array([3.66551554,   7.18780739, -11.10346907]))  # check position of random cp_point
        # assert_almost_equal(cp_points[49], np.array([3.66551554,   7.18780739,  11.10346907]))  # check position of random cp_point

        cp_points_above_water_z_mask, _ = extract_above_water_quantities(cp_points, cp_points)
        assert all(cp_points_above_water_z_mask[:, 2] > 0)  # all z-coordinates are positive, i.e. above water

        cp_jib_reference = sail_set.sails[0].get_cp_points1d()
        cp_points_above_water_jib_reference, _ = extract_above_water_quantities(cp_jib_reference, cp_jib_reference)
        cp_points_above_water_jib_extracted = sail_set.extract_data_above_water_by_id(cp_points, 0)
        assert_almost_equal(cp_points_above_water_jib_reference, cp_points_above_water_jib_extracted)

        cp_mainsail_reference = sail_set.sails[1].get_cp_points1d()
        cp_points_above_water_mainsail_reference, _ = extract_above_water_quantities(cp_mainsail_reference, cp_mainsail_reference)
        cp_points_above_water_mainsail_extracted = sail_set.extract_data_above_water_by_id(cp_points, 1)
        assert_almost_equal(cp_points_above_water_mainsail_reference, cp_points_above_water_mainsail_extracted)

        # cp_points_above_water_extracted = np.hstack([cp_points_above_water_jib_extracted, cp_points_above_water_mainsail_extracted])
        # assert_almost_equal(cp_points_above_water_z_mask, cp_points_above_water_extracted)


    def test_sail_set_rotations(self):
        sail_set = self._prepare_sail_set(initial_twist_factor=0.)
        le_mid_points, cp_points, ctr_points, te_mid_points = _prepare_geometry_data_to_display(sail_set.panels1d)

        le_mid_points_desired = np.array(
            [[1.28850858, 4.70493707, -7.63784408],
             [0.78082695, 4.20434279, -6.93665599],
             [0.27314532, 3.70374852, -6.2354679],
             [-0.2345363, 3.20315425, -5.53427981],
             [-0.74221793, 2.70255997, -4.83309172],
             [-1.24989956, 2.2019657, -4.13190363],
             [-1.75758118, 1.70137142, -3.43071554],
             [-2.26526281, 1.20077715, -2.72952744],
             [-2.77294444, 0.70018287, -2.02833935],
             [-3.28062606, 0.1995886, -1.32715126],
             [-3.28062606, 0.1995886, 1.32715126],
             [-2.77294444, 0.70018287, 2.02833935],
             [-2.26526281, 1.20077715, 2.72952744],
             [-1.75758118, 1.70137142, 3.43071554],
             [-1.24989956, 2.2019657, 4.13190363],
             [-0.74221793, 2.70255997, 4.83309172],
             [-0.2345363, 3.20315425, 5.53427981],
             [0.27314532, 3.70374852, 6.2354679],
             [0.78082695, 4.20434279, 6.93665599],
             [1.28850858, 4.70493707, 7.63784408],
             [3.64477198, 7.45558034, -11.62103089],
             [3.32827918, 6.80817701, -10.61192176],
             [3.01178637, 6.16077367, -9.60281264],
             [2.69529357, 5.51337034, -8.59370352],
             [2.37880076, 4.865967, -7.58459439],
             [2.06230796, 4.21856367, -6.57548527],
             [1.74581515, 3.57116033, -5.56637614],
             [1.42932235, 2.923757, -4.55726702],
             [1.11282954, 2.27635366, -3.54815789],
             [0.79633674, 1.62895033, -2.53904877],
             [0.79633674, 1.62895033, 2.53904877],
             [1.11282954, 2.27635366, 3.54815789],
             [1.42932235, 2.923757, 4.55726702],
             [1.74581515, 3.57116033, 5.56637614],
             [2.06230796, 4.21856367, 6.57548527],
             [2.37880076, 4.865967, 7.58459439],
             [2.69529357, 5.51337034, 8.59370352],
             [3.01178637, 6.16077367, 9.60281264],
             [3.32827918, 6.80817701, 10.61192176],
             [3.64477198, 7.45558034, 11.62103089]])

        cp_points_desired = np.array(
            [[1.45247907, 4.73384949, -7.63784408],
             [1.02653648, 4.24766801, -6.93665599],
             [0.6003477, 3.76144313, -6.2354679],
             [0.17366651, 3.27513142, -5.53427981],
             [-0.25326088, 2.78877629, -4.83309172],
             [-0.67969587, 2.30250799, -4.13190363],
             [-1.10563845, 1.81632652, -3.43071554],
             [-1.53182724, 1.33010163, -2.72952744],
             [-1.95850843, 0.84378992, -2.02833935],
             [-2.38543582, 0.35743479, -1.32715126],
             [-2.38543582, 0.35743479, 1.32715126],
             [-1.95850843, 0.84378992, 2.02833935],
             [-1.53182724, 1.33010163, 2.72952744],
             [-1.10563845, 1.81632652, 3.43071554],
             [-0.67969587, 2.30250799, 4.13190363],
             [-0.25326088, 2.78877629, 4.83309172],
             [0.17366651, 3.27513142, 5.53427981],
             [0.6003477, 3.76144313, 6.2354679],
             [1.02653648, 4.24766801, 6.93665599],
             [1.45247907, 4.73384949, 7.63784408],
             [4.16868971, 7.54796117, -11.62103089],
             [3.9152246, 6.91167132, -10.61192176],
             [3.65978987, 6.27503417, -9.60281264],
             [3.40041592, 5.63770243, -8.59370352],
             [3.13907235, 5.0000234, -7.58459439],
             [2.87181993, 4.36130247, -6.57548527],
             [2.59865867, 3.72153965, -5.56637614],
             [2.32352779, 3.08142954, -4.55726702],
             [2.04445768, 2.44062484, -3.54815789],
             [1.76341795, 1.79947284, -2.53904877],
             [1.76341795, 1.79947284, 2.53904877],
             [2.04445768, 2.44062484, 3.54815789],
             [2.32352779, 3.08142954, 4.55726702],
             [2.59865867, 3.72153965, 5.56637614],
             [2.87181993, 4.36130247, 6.57548527],
             [3.13907235, 5.0000234, 7.58459439],
             [3.40041592, 5.63770243, 8.59370352],
             [3.65978987, 6.27503417, 9.60281264],
             [3.9152246, 6.91167132, 10.61192176],
             [4.16868971, 7.54796117, 11.62103089]])

        ctr_points_desired = np.array(
            [[1.78042005, 4.79167433, -7.63784408],
             [1.51795555, 4.33431845, -6.93665599],
             [1.25475245, 3.87683234, -6.2354679],
             [0.99007214, 3.41908575, -5.53427981],
             [0.72465322, 2.96120893, -4.83309172],
             [0.46071151, 2.50359258, -4.13190363],
             [0.19824701, 2.0462367, -3.43071554],
             [-0.06495609, 1.58875059, -2.72952744],
             [-0.3296364, 1.131004, -2.02833935],
             [-0.59505532, 0.67312718, -1.32715126],
             [-0.59505532, 0.67312718, 1.32715126],
             [-0.3296364, 1.131004, 2.02833935],
             [-0.06495609, 1.58875059, 2.72952744],
             [0.19824701, 2.0462367, 3.43071554],
             [0.46071151, 2.50359258, 4.13190363],
             [0.72465322, 2.96120893, 4.83309172],
             [0.99007214, 3.41908575, 5.53427981],
             [1.25475245, 3.87683234, 6.2354679],
             [1.51795555, 4.33431845, 6.93665599],
             [1.78042005, 4.79167433, 7.63784408],
             [5.21652516, 7.73272283, -11.62103089],
             [5.08911544, 7.11865995, -10.61192176],
             [4.95579688, 6.50355518, -9.60281264],
             [4.81066062, 5.88636662, -8.59370352],
             [4.65961552, 5.26813618, -7.58459439],
             [4.49084388, 4.64678007, -6.57548527],
             [4.30434569, 4.0222983, -5.56637614],
             [4.11193867, 3.39677463, -4.55726702],
             [3.90771394, 2.76916719, -3.54815789],
             [3.69758038, 2.14051786, -2.53904877],
             [3.69758038, 2.14051786, 2.53904877],
             [3.90771394, 2.76916719, 3.54815789],
             [4.11193867, 3.39677463, 4.55726702],
             [4.30434569, 4.0222983, 5.56637614],
             [4.49084388, 4.64678007, 6.57548527],
             [4.65961552, 5.26813618, 7.58459439],
             [4.81066062, 5.88636662, 8.59370352],
             [4.95579688, 6.50355518, 9.60281264],
             [5.08911544, 7.11865995, 10.61192176],
             [5.21652516, 7.73272283, 11.62103089]])

        te_mid_points_desired = np.array(
            [[1.94439054, 4.82058675, -7.63784408],
             [1.76366509, 4.37764368, -6.93665599],
             [1.58195483, 3.93452695, -6.2354679],
             [1.39827495, 3.49106292, -5.53427981],
             [1.21361027, 3.04742525, -4.83309172],
             [1.0309152, 2.60413488, -4.13190363],
             [0.85018975, 2.1611918, -3.43071554],
             [0.66847949, 1.71807507, -2.72952744],
             [0.48479961, 1.27461105, -2.02833935],
             [0.30013493, 0.83097337, -1.32715126],
             [0.30013493, 0.83097337, 1.32715126],
             [0.48479961, 1.27461105, 2.02833935],
             [0.66847949, 1.71807507, 2.72952744],
             [0.85018975, 2.1611918, 3.43071554],
             [1.0309152, 2.60413488, 4.13190363],
             [1.21361027, 3.04742525, 4.83309172],
             [1.39827495, 3.49106292, 5.53427981],
             [1.58195483, 3.93452695, 6.2354679],
             [1.76366509, 4.37764368, 6.93665599],
             [1.94439054, 4.82058675, 7.63784408],
             [5.74044288, 7.82510367, -11.62103089],
             [5.67606086, 7.22215426, -10.61192176],
             [5.60380038, 6.61781568, -9.60281264],
             [5.51578297, 6.01069872, -8.59370352],
             [5.4198871, 5.40219258, -7.58459439],
             [5.30035585, 4.78951888, -6.57548527],
             [5.15718921, 4.17267762, -5.56637614],
             [5.00614411, 3.55444718, -4.55726702],
             [4.83934208, 2.93343837, -3.54815789],
             [4.66466159, 2.31104037, -2.53904877],
             [4.66466159, 2.31104037, 2.53904877],
             [4.83934208, 2.93343837, 3.54815789],
             [5.00614411, 3.55444718, 4.55726702],
             [5.15718921, 4.17267762, 5.56637614],
             [5.30035585, 4.78951888, 6.57548527],
             [5.4198871, 5.40219258, 7.58459439],
             [5.51578297, 6.01069872, 8.59370352],
             [5.60380038, 6.61781568, 9.60281264],
             [5.67606086, 7.22215426, 10.61192176],
             [5.74044288, 7.82510367, 11.62103089]])

        assert_almost_equal(le_mid_points, le_mid_points_desired)
        assert_almost_equal(cp_points, cp_points_desired)
        assert_almost_equal(ctr_points, ctr_points_desired)
        assert_almost_equal(te_mid_points, te_mid_points_desired)

    def test_sail_set_rotations_with_initial_sail_twist(self):
        sail_set = self._prepare_sail_set(initial_twist_factor=10.)

        le_mid_points, cp_points, ctr_points, te_mid_points = _prepare_geometry_data_to_display(sail_set.panels1d)

        cp_points_desired = np.array(
            [[1.44764504, 4.75284625, -7.62778184],
             [1.02023791, 4.27320956, -6.92298165],
             [0.59318227, 3.7914796, -6.21921217],
             [0.16618365, 3.30760366, -5.51651494],
             [-0.26057091, 2.82166269, -4.81490605],
             [-0.68640451, 2.33384068, -4.11439176],
             [-1.11136527, 1.84413196, -3.41501099],
             [-1.53623428, 1.35237154, -2.71681928],
             [-1.96131051, 0.85852802, -2.01984628],
             [-2.38640347, 0.36267309, -1.32411213],
             [-2.38640347, 0.36267309, 1.32411213],
             [-1.96131051, 0.85852802, 2.01984628],
             [-1.53623428, 1.35237154, 2.71681928],
             [-1.11136527, 1.84413196, 3.41501099],
             [-0.68640451, 2.33384068, 4.11439176],
             [-0.26057091, 2.82166269, 4.81490605],
             [0.16618365, 3.30760366, 5.51651494],
             [0.59318227, 3.7914796, 6.21921217],
             [1.02023791, 4.27320956, 6.92298165],
             [1.44764504, 4.75284625, 7.62778184],
             [4.14979607, 7.61814373, -11.58193034],
             [3.89695638, 6.98208129, -10.57247916],
             [3.64264607, 6.34368336, -9.56414703],
             [3.3848663, 5.70248709, -8.55701724],
             [3.12545572, 5.05914419, -7.55093555],
             [2.86044474, 4.41284686, -6.54598415],
             [2.58974627, 3.76376277, -5.54208275],
             [2.31715587, 3.11303491, -4.53898876],
             [2.040661, 2.46034622, -3.53669623],
             [1.76216332, 1.80621491, -2.53511682],
             [1.76216332, 1.80621491, 2.53511682],
             [2.040661, 2.46034622, 3.53669623],
             [2.31715587, 3.11303491, 4.53898876],
             [2.58974627, 3.76376277, 5.54208275],
             [2.86044474, 4.41284686, 6.54598415],
             [3.12545572, 5.05914419, 7.55093555],
             [3.3848663, 5.70248709, 8.55701724],
             [3.64264607, 6.34368336, 9.56414703],
             [3.89695638, 6.98208129, 10.57247916],
             [4.14979607, 7.61814373, 11.58193034]])

        assert_almost_equal(-cp_points[0][2], cp_points[19][2])
        assert_almost_equal(-cp_points[20][2], cp_points[39][2])
        assert_almost_equal(cp_points, cp_points_desired)
