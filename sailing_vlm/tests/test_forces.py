import numpy as np

from sailing_vlm.solver.coefs import calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points, \
                                            calc_velocity_coefs, \
                                            solve_eq, \
                                            get_vlm_CL_CD_free_wing, \
                                            get_CL_CD_free_wing, calculate_RHS

from sailing_vlm.solver.panels import get_panels_area, make_panels_from_le_te_points
from sailing_vlm.solver.forces import determine_vector_from_its_dot_and_cross_product, \
    calc_pressure, calc_moment_arm_in_shifted_csys
from sailing_vlm.solver.velocity import calculate_app_fs
from sailing_vlm.solver.forces import is_no_flux_BC_satisfied, calc_force_wrapper
from sailing_vlm.rotations.geometry_calc import rotation_matrix
from unittest import TestCase

class TestForces(TestCase):
    def setUp(self):
        
        self.gamma_orientation = 1
        ### WING DEFINITION ###
        # Parameters #
        chord = 1.  # chord length
        half_wing_span = 100.  # wing span length

        # Points defining wing (x,y,z) #
        self.le_NW = np.array([0., half_wing_span, 0.])  # leading edge North - West coordinate
        self.le_SW = np.array([0., -half_wing_span, 0.])  # leading edge South - West coordinate

        self.te_NE = np.array([chord, half_wing_span, 0.])  # trailing edge North - East coordinate
        self.te_SE = np.array([chord, -half_wing_span, 0.])  # trailing edge South - East coordinate

        AoA_deg = 3.0  # Angle of attack [deg]
        self.Ry = rotation_matrix([0, 1, 0], np.deg2rad(AoA_deg))

        # reference values - to compare with book coeff_formulas
        self.AR = 2 * half_wing_span / chord
        self.S = 2 * half_wing_span * chord
        self.CL_expected, self.CD_ind_expected = get_CL_CD_free_wing(self.AR, AoA_deg)

        ### FLIGHT CONDITIONS ###
        self.V = [10.0, 0.0, 0.0]
        self.rho = 1.225  # fluid density [kg/m3]


    def get_geom(self,ns,nc):
        panels, trailing_edge_info, leading_edge_info = make_panels_from_le_te_points(

            [np.dot(self.Ry, self.le_SW),
             np.dot(self.Ry, self.te_SE),
             np.dot(self.Ry, self.le_NW),
             np.dot(self.Ry, self.te_NE)],
            [nc, ns])

        return panels, trailing_edge_info, leading_edge_info

    def test_CL_CD_spanwise_only(self):
        ### ARRANGE ###
        ### MESH DENSITY ###
        ns = 20  # number of panels (spanwise)
        nc = 1   # number of panels (chordwise)
        N = ns * nc

        panels, trailing_edge_info, leading_edge_info = self.get_geom(ns, nc)
        V_app_infw = np.array([self.V for _ in range(N)])

        ### ACT ###
        ### CALCULATIONS ###
        areas = get_panels_area(panels) 
        normals, ctr_p, center_of_pressure, rings, span_vectors, _, _ = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels, self.gamma_orientation)

        coefs, v_ind_coeff = calc_velocity_coefs(V_app_infw, ctr_p, rings, normals, trailing_edge_info, self.gamma_orientation)
        RHS = calculate_RHS(V_app_infw, normals)
        gamma_magnitude = solve_eq(coefs, RHS)

        _,  V_app_fs_at_ctrl_p = calculate_app_fs(V_app_infw,  v_ind_coeff,  gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, panels, areas, normals)

        force, _, _ = calc_force_wrapper(V_app_infw, gamma_magnitude, self.rho, center_of_pressure, rings, ns, normals, span_vectors, trailing_edge_info, leading_edge_info, self.gamma_orientation)

        CL_vlm, CD_vlm = get_vlm_CL_CD_free_wing(force, self.V, self.rho, self.S)

        ### ASSSERT ###
        np.testing.assert_almost_equal(CL_vlm, 0.32477746534138485, decimal=5)
        np.testing.assert_almost_equal(CD_vlm, 0.00020242110304907, decimal=5)

    def test_CL_CD_spanwise_and_chordwise(self):
        ### ARRANGE ###
        ### MESH DENSITY ###
        ns = 20  # number of panels (spanwise)
        nc = 3  # number of panels (chordwise)
        N = ns*nc

        panels, trailing_edge_info, leading_edge_info = self.get_geom(ns, nc)
        V_app_infw = np.array([self.V for _ in range(N)])

        ### ACT ###
        ### CALCULATIONS ###
        areas = get_panels_area(panels) 
        normals, ctr_p, center_of_pressure, rings, span_vectors, _, _ = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels, self.gamma_orientation)

        coefs, v_ind_coeff = calc_velocity_coefs(V_app_infw, ctr_p, rings, normals, trailing_edge_info, self.gamma_orientation)
        RHS = calculate_RHS(V_app_infw, normals)
        gamma_magnitude = solve_eq(coefs, RHS)

        _,  V_app_fs_at_ctrl_p = calculate_app_fs(V_app_infw,  v_ind_coeff,  gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, panels, areas, normals)

        force, _, _ = calc_force_wrapper(V_app_infw, gamma_magnitude, self.rho, center_of_pressure, rings, ns, normals, span_vectors, trailing_edge_info, leading_edge_info, self.gamma_orientation)

        CL_vlm, CD_vlm = get_vlm_CL_CD_free_wing(force, self.V, self.rho, self.S)

        ### ASSSERT ###
        np.testing.assert_almost_equal(CL_vlm, 0.3247765909739283, decimal=5)
        np.testing.assert_almost_equal(CD_vlm, 0.0002024171446522)


    def test_determine_vector_from_its_dot_and_cross_product(self):
        F = np.array([[-5.69486970e+00, 1.89981982e+02, -7.51769175e+01],
                      [-4.10503550e+01, 2.41966384e+02, -8.68481233e+01],
                      [-6.37389199e+01, 2.71344824e+02, -9.27215801e+01],
                      [-7.81465247e+01, 2.91970809e+02, -9.72597369e+01],
                      [-9.07817814e+01, 3.02213594e+02, -9.80102392e+01],
                      [-1.01282827e+02, 3.03842433e+02, -9.57870102e+01],
                      [-1.09256106e+02, 2.98521471e+02, -9.13828534e+01],
                      [-1.14441029e+02, 2.87721219e+02, -8.54828061e+01],
                      [-1.16757444e+02, 2.72715947e+02, -7.86449205e+01],
                      [-1.16291044e+02, 2.54610484e+02, -7.13116309e+01],
                      [-1.13264945e+02, 2.34365721e+02, -6.38257769e+01],
                      [-1.08006936e+02, 2.12818001e+02, -5.64464629e+01],
                      [-1.00909715e+02, 1.90694662e+02, -4.93663086e+01],
                      [-9.23997691e+01, 1.68555001e+02, -4.26971559e+01],
                      [-8.29063690e+01, 1.46629127e+02, -3.64114583e+01],
                      [-7.27363402e+01, 1.25120698e+02, -3.05029389e+01],
                      [-6.21550967e+01, 1.04267462e+02, -2.49897555e+01],
                      [-5.14305959e+01, 8.43354039e+01, -1.98999234e+01],
                      [-4.08484596e+01, 6.56207612e+01, -1.52683671e+01],
                      [-3.07206832e+01, 4.84535306e+01, -1.11365432e+01],
                      [-2.13911995e+01, 3.32011476e+01, -7.55315966e+00],
                      [-1.32385826e+01, 2.02714611e+01, -4.57554664e+00],
                      [-6.67455403e+00, 1.01135982e+01, -2.27144612e+00],
                      [-2.13496399e+00, 3.21424845e+00, -7.21048369e-01],
                      [-2.82159144e-02, 4.24992393e-02, -9.57982199e-03],
                      [-1.19377340e+01, 1.61498095e+02, -6.64646854e+01],
                      [-2.35424616e+01, 2.08269882e+02, -8.57603690e+01],
                      [-2.99505116e+01, 2.34198656e+02, -9.64237620e+01],
                      [-3.67892124e+01, 2.47467381e+02, -1.01904512e+02],
                      [-4.22888518e+01, 2.56888537e+02, -1.05792823e+02],
                      [-4.80347532e+01, 2.63048875e+02, -1.08357737e+02],
                      [-5.40089320e+01, 2.66897268e+02, -1.09990804e+02],
                      [-6.01141037e+01, 2.68911467e+02, -1.10888219e+02],
                      [-6.61877807e+01, 2.69274120e+02, -1.11123367e+02],
                      [-7.20224319e+01, 2.68003907e+02, -1.10700906e+02],
                      [-7.74113709e+01, 2.65056044e+02, -1.09598815e+02],
                      [-8.22031744e+01, 2.60392036e+02, -1.07798197e+02],
                      [-8.62905722e+01, 2.53980968e+02, -1.05283683e+02],
                      [-8.95451845e+01, 2.45792169e+02, -1.02038524e+02],
                      [-9.17475291e+01, 2.35733413e+02, -9.80246483e+01],
                      [-9.25532408e+01, 2.23587664e+02, -9.31519076e+01],
                      [-9.15056023e+01, 2.09049937e+02, -8.72867162e+01],
                      [-8.81713435e+01, 1.91844469e+02, -8.03011633e+01],
                      [-8.23615815e+01, 1.71885363e+02, -7.21447694e+01],
                      [-7.40765463e+01, 1.49251317e+02, -6.28340572e+01],
                      [-6.26230944e+01, 1.23707093e+02, -5.22200384e+01],
                      [-4.91988202e+01, 9.62414807e+01, -4.07252522e+01],
                      [-3.60038827e+01, 6.95269667e+01, -2.95091655e+01],
                      [-2.33784141e+01, 4.45303246e+01, -1.89654881e+01],
                      [-1.16460782e+01, 2.18737790e+01, -9.35274112e+00]])

        r = np.array([[-4.5126404, 0.87308483, 1.36378624],
                      [-4.32423872, 1.15434184, 2.05914957],
                      [-4.13369845, 1.43643024, 2.7542652],
                      [-3.94119514, 1.71928179, 3.44915345],
                      [-3.74690434, 2.00282825, 4.14383466],
                      [-3.55100159, 2.28700136, 4.83832917],
                      [-3.35366244, 2.57173288, 5.5326573],
                      [-3.15506244, 2.85695457, 6.2268394],
                      [-2.95537714, 3.14259818, 6.92089578],
                      [-2.75478207, 3.42859546, 7.6148468],
                      [-2.55345279, 3.71487818, 8.30871277],
                      [-2.35156484, 4.00137808, 9.00251403],
                      [-2.14929467, 4.28802658, 9.69627102],
                      [-1.94684005, 4.57474678, 10.39000665],
                      [-1.7444397, 4.86144589, 11.08374856],
                      [-1.54235367, 5.1480228, 11.77752688],
                      [-1.34084292, 5.43437606, 12.47137183],
                      [-1.14016838, 5.72040424, 13.16531364],
                      [-0.94059101, 6.00600589, 13.85938252],
                      [-0.74237175, 6.29107956, 14.55360872],
                      [-0.54577156, 6.57552381, 15.24802244],
                      [-0.35105138, 6.85923719, 15.94265392],
                      [-0.15847216, 7.14211825, 16.63753338],
                      [0.03170515, 7.42406555, 17.33269105],
                      [0.21921961, 7.70497765, 18.02815714],
                      [1.76355604, 2.05465732, 3.34320208],
                      [1.75335351, 2.36432784, 4.09778078],
                      [1.74234257, 2.67373086, 4.85244998],
                      [1.73052804, 2.98286797, 5.60720914],
                      [1.71791477, 3.29174078, 6.36205773],
                      [1.70450758, 3.60035089, 7.11699519],
                      [1.6903113, 3.9086999, 7.87202099],
                      [1.67533078, 4.2167894, 8.62713459],
                      [1.65957085, 4.52462099, 9.38233545],
                      [1.64303634, 4.83219628, 10.13762302],
                      [1.62573207, 5.13951687, 10.89299677],
                      [1.6076629, 5.44658435, 11.64845615],
                      [1.58882911, 5.75339882, 12.40400113],
                      [1.56912226, 6.05992439, 13.15964385],
                      [1.54822544, 6.36605621, 13.91541979],
                      [1.525713, 6.67165343, 14.6713766],
                      [1.50115475, 6.9765737, 15.42756244],
                      [1.4741205, 7.28067467, 16.18402548],
                      [1.44418006, 7.583814, 16.94081386],
                      [1.41090324, 7.88584933, 17.69797576],
                      [1.37385986, 8.18663832, 18.45555934],
                      [1.33261972, 8.48603862, 19.21361274],
                      [1.28675264, 8.78390788, 19.97218414],
                      [1.23582844, 9.08010377, 20.7313217],
                      [1.17941691, 9.37448392, 21.49107356]])

        # F_total = np.array([-3119.88053533,  9169.50368074, -3314.94364126])
        # M_total_expected = np.array([-97190.75616098, -32915.36423537, 10338.34931543])

        r_dot_F = np.array([np.dot(r[i], F[i]) for i in range(len(F))])
        r_cross_F = np.cross(r, F)  # moments

        F_total = np.sum(F, axis=0)  # total force
        r_dot_F_total = np.sum(r_dot_F, axis=0)
        r_cross_F_total = np.sum(r_cross_F, axis=0)  # total moments

        np.testing.assert_almost_equal(r_cross_F_total, np.array([-97190.75616098, -32915.36423537, 10338.34931543]), decimal=4)

        R_estimate = determine_vector_from_its_dot_and_cross_product(F_total, r_dot_F_total, r_cross_F_total)
        np.testing.assert_almost_equal(np.dot(R_estimate, F_total), r_dot_F_total, decimal=4)

        R_estimate_cross_F_total = np.cross(R_estimate, F_total)
        R_estimate2 = determine_vector_from_its_dot_and_cross_product(F_total, r_dot_F_total, R_estimate_cross_F_total)
        np.testing.assert_almost_equal(R_estimate, R_estimate2, decimal=4)

        np.testing.assert_almost_equal(R_estimate_cross_F_total, np.array([-98169.12235103, -30039.89115736,   9298.81310178]), decimal=4)

        # most naive way
        # r0 = r_cross_F_total[0] / F_total[0]
        # r1 = r_cross_F_total[1] / F_total[1]
        # r2 = r_cross_F_total[2] / F_total[2]
        # r_naive = np.array([r0, r1, r2])
        # M_naive = np.cross(r_naive, F_total)
        #todo: this is weird: r_cross_F_total != R_estimate_cross_F_total


    def test_calc_moment_arm_in_shifted_csys(self):
        
        cp = np.array([[ -2.85932546,   0.40627935,   3.48110962],
                                        [ -1.38806442,   1.33606872,   8.09527974],
                                        [ -1.60224763,   0.67329836,   3.45805989],
                                        [ -0.94606188,   1.47100623,   8.08232929],
                                        [  0.54484912,   1.09780989,   5.49870887],
                                        [  0.53630003,   2.18727047,  11.59919224],
                                        [  2.28427654,   1.54835841,   5.44728884],
                                        [  1.75442245,   2.63275344,  11.54135142],
                                        [ -1.38806442,   1.33606872,  -8.09527974],
                                        [ -2.85932546,   0.40627935,  -3.48110962],
                                        [ -0.94606188,   1.47100623,  -8.08232929],
                                        [ -1.60224763,   0.67329836,  -3.45805989],
                                        [  0.53630003,   2.18727047, -11.59919224],
                                        [  0.54484912,   1.09780989,  -5.49870887],
                                        [  1.75442245,   2.63275344, -11.54135142],
                                        [  2.28427654,   1.54835841,  -5.44728884]])
        v_from_original_xyz_2_reference_csys_xyz = np.array([0, 0, 0])
        
        r_expected = np.array([[ -2.85932546,   0.40627935,   3.48110962],
                                [ -1.38806442,   1.33606872,   8.09527974],
                                [ -1.60224763,   0.67329836,   3.45805989],
                                [ -0.94606188,   1.47100623,   8.08232929],
                                [  0.54484912,   1.09780989,   5.49870887],
                                [  0.53630003,   2.18727047,  11.59919224],
                                [  2.28427654,   1.54835841,   5.44728884],
                                [  1.75442245,   2.63275344,  11.54135142],
                                [ -1.38806442,   1.33606872,  -8.09527974],
                                [ -2.85932546,   0.40627935,  -3.48110962],
                                [ -0.94606188,   1.47100623,  -8.08232929],
                                [ -1.60224763,   0.67329836,  -3.45805989],
                                [  0.53630003,   2.18727047, -11.59919224],
                                [  0.54484912,   1.09780989,  -5.49870887],
                                [  1.75442245,   2.63275344, -11.54135142],
                                [  2.28427654,   1.54835841,  -5.44728884]])
        r = calc_moment_arm_in_shifted_csys(cp, v_from_original_xyz_2_reference_csys_xyz)            
        np.testing.assert_almost_equal(r, r_expected)
        
    def test_calc_force_wrapper(self):
        
        ns = 5
        nc = 2
        panels = np.array([[[ 4.99314767e-01, -1.00000000e+02, -2.61679781e-02],
                            [ 0.00000000e+00, -1.00000000e+02,  0.00000000e+00],
                            [ 0.00000000e+00, -6.00000000e+01,  0.00000000e+00],
                            [ 4.99314767e-01, -6.00000000e+01, -2.61679781e-02]],

                            [[ 4.99314767e-01, -6.00000000e+01, -2.61679781e-02],
                            [ 0.00000000e+00, -6.00000000e+01,  0.00000000e+00],
                            [ 0.00000000e+00, -2.00000000e+01,  0.00000000e+00],
                            [ 4.99314767e-01, -2.00000000e+01, -2.61679781e-02]],

                            [[ 4.99314767e-01, -2.00000000e+01, -2.61679781e-02],
                            [ 0.00000000e+00, -2.00000000e+01,  0.00000000e+00],
                            [ 0.00000000e+00,  2.00000000e+01,  0.00000000e+00],
                            [ 4.99314767e-01,  2.00000000e+01, -2.61679781e-02]],

                            [[ 4.99314767e-01,  2.00000000e+01, -2.61679781e-02],
                            [ 0.00000000e+00,  2.00000000e+01,  0.00000000e+00],
                            [ 0.00000000e+00,  6.00000000e+01,  0.00000000e+00],
                            [ 4.99314767e-01,  6.00000000e+01, -2.61679781e-02]],

                            [[ 4.99314767e-01,  6.00000000e+01, -2.61679781e-02],
                            [ 0.00000000e+00,  6.00000000e+01,  0.00000000e+00],
                            [ 0.00000000e+00,  1.00000000e+02,  0.00000000e+00],
                            [ 4.99314767e-01,  1.00000000e+02, -2.61679781e-02]],

                            [[ 9.98629535e-01, -1.00000000e+02, -5.23359562e-02],
                            [ 4.99314767e-01, -1.00000000e+02, -2.61679781e-02],
                            [ 4.99314767e-01, -6.00000000e+01, -2.61679781e-02],
                            [ 9.98629535e-01, -6.00000000e+01, -5.23359562e-02]],

                            [[ 9.98629535e-01, -6.00000000e+01, -5.23359562e-02],
                            [ 4.99314767e-01, -6.00000000e+01, -2.61679781e-02],
                            [ 4.99314767e-01, -2.00000000e+01, -2.61679781e-02],
                            [ 9.98629535e-01, -2.00000000e+01, -5.23359562e-02]],

                            [[ 9.98629535e-01, -2.00000000e+01, -5.23359562e-02],
                            [ 4.99314767e-01, -2.00000000e+01, -2.61679781e-02],
                            [ 4.99314767e-01,  2.00000000e+01, -2.61679781e-02],
                            [ 9.98629535e-01,  2.00000000e+01, -5.23359562e-02]],

                            [[ 9.98629535e-01,  2.00000000e+01, -5.23359562e-02],
                            [ 4.99314767e-01,  2.00000000e+01, -2.61679781e-02],
                            [ 4.99314767e-01,  6.00000000e+01, -2.61679781e-02],
                            [ 9.98629535e-01,  6.00000000e+01, -5.23359562e-02]],

                            [[ 9.98629535e-01,  6.00000000e+01, -5.23359562e-02],
                            [ 4.99314767e-01,  6.00000000e+01, -2.61679781e-02],
                            [ 4.99314767e-01,  1.00000000e+02, -2.61679781e-02],
                            [ 9.98629535e-01,  1.00000000e+02, -5.23359562e-02]]])
        areas = np.array([20., 20., 20., 20., 20., 20., 20., 20., 20., 20.])   
        normals = np.array([[0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953],
                            [0.05233596, 0.        , 0.99862953]])   

        V_app_infw = np.array([ [10.,  0.,  0.],
                                [10.,  0.,  0.],
                                [10.,  0.,  0.],
                                [10.,  0.,  0.],
                                [10.,  0.,  0.],
                                [10.,  0.,  0.],
                                [10.,  0.,  0.],
                                [10.,  0.,  0.],
                                [10.,  0.,  0.],
                                [10.,  0.,  0.]])
        gamma_magnitude = np.array([1.21622984, 1.2257545 , 1.22694307, 1.2257545 , 1.21622984,
                                    1.62157652, 1.63433059, 1.63591882, 1.63433059, 1.62157652])
        
        trailing_edge_info = np.array([False, False, False, False, False,  True,  True,  True,  True, True])
        leading_edge_info = np.array([ True,  True,  True,  True,  True, False, False, False, False,False])
        
        span_vectors = np.array([[ 0., 40.,  0.],
                                [ 0., 40.,  0.],
                                [ 0., 40.,  0.],
                                [ 0., 40.,  0.],
                                [ 0., 40.,  0.],
                                [ 0., 40.,  0.],
                                [ 0., 40.,  0.],
                                [ 0., 40.,  0.],
                                [ 0., 40.,  0.],
                                [ 0., 40.,  0.]])
        
        cp = np.array([[ 1.24828692e-01, -8.00000000e+01, -6.54199453e-03],
                                        [ 1.24828692e-01, -4.00000000e+01, -6.54199453e-03],
                                        [ 1.24828692e-01, -7.10542736e-15, -6.54199453e-03],
                                        [ 1.24828692e-01,  4.00000000e+01, -6.54199453e-03],
                                        [ 1.24828692e-01,  8.00000000e+01, -6.54199453e-03],
                                        [ 6.24143459e-01, -8.00000000e+01, -3.27099727e-02],
                                        [ 6.24143459e-01, -4.00000000e+01, -3.27099727e-02],
                                        [ 6.24143459e-01, -7.10542736e-15, -3.27099727e-02],
                                        [ 6.24143459e-01,  4.00000000e+01, -3.27099727e-02],
                                        [ 6.24143459e-01,  8.00000000e+01, -3.27099727e-02]])
        
        rings = np.array([[[ 6.24143459e-01, -1.00000000e+02, -3.27099727e-02],
                            [ 1.24828692e-01, -1.00000000e+02, -6.54199453e-03],
                            [ 1.24828692e-01, -6.00000000e+01, -6.54199453e-03],
                            [ 6.24143459e-01, -6.00000000e+01, -3.27099727e-02]],

                            [[ 6.24143459e-01, -6.00000000e+01, -3.27099727e-02],
                            [ 1.24828692e-01, -6.00000000e+01, -6.54199453e-03],
                            [ 1.24828692e-01, -2.00000000e+01, -6.54199453e-03],
                            [ 6.24143459e-01, -2.00000000e+01, -3.27099727e-02]],

                            [[ 6.24143459e-01, -2.00000000e+01, -3.27099727e-02],
                            [ 1.24828692e-01, -2.00000000e+01, -6.54199453e-03],
                            [ 1.24828692e-01,  2.00000000e+01, -6.54199453e-03],
                            [ 6.24143459e-01,  2.00000000e+01, -3.27099727e-02]],

                            [[ 6.24143459e-01,  2.00000000e+01, -3.27099727e-02],
                            [ 1.24828692e-01,  2.00000000e+01, -6.54199453e-03],
                            [ 1.24828692e-01,  6.00000000e+01, -6.54199453e-03],
                            [ 6.24143459e-01,  6.00000000e+01, -3.27099727e-02]],

                            [[ 6.24143459e-01,  6.00000000e+01, -3.27099727e-02],
                            [ 1.24828692e-01,  6.00000000e+01, -6.54199453e-03],
                            [ 1.24828692e-01,  1.00000000e+02, -6.54199453e-03],
                            [ 6.24143459e-01,  1.00000000e+02, -3.27099727e-02]],

                            [[ 1.12345823e+00, -1.00000000e+02, -5.88779508e-02],
                            [ 6.24143459e-01, -1.00000000e+02, -3.27099727e-02],
                            [ 6.24143459e-01, -6.00000000e+01, -3.27099727e-02],
                            [ 1.12345823e+00, -6.00000000e+01, -5.88779508e-02]],

                            [[ 1.12345823e+00, -6.00000000e+01, -5.88779508e-02],
                            [ 6.24143459e-01, -6.00000000e+01, -3.27099727e-02],
                            [ 6.24143459e-01, -2.00000000e+01, -3.27099727e-02],
                            [ 1.12345823e+00, -2.00000000e+01, -5.88779508e-02]],

                            [[ 1.12345823e+00, -2.00000000e+01, -5.88779508e-02],
                            [ 6.24143459e-01, -2.00000000e+01, -3.27099727e-02],
                            [ 6.24143459e-01,  2.00000000e+01, -3.27099727e-02],
                            [ 1.12345823e+00,  2.00000000e+01, -5.88779508e-02]],

                            [[ 1.12345823e+00,  2.00000000e+01, -5.88779508e-02],
                            [ 6.24143459e-01,  2.00000000e+01, -3.27099727e-02],
                            [ 6.24143459e-01,  6.00000000e+01, -3.27099727e-02],
                            [ 1.12345823e+00,  6.00000000e+01, -5.88779508e-02]],

                            [[ 1.12345823e+00,  6.00000000e+01, -5.88779508e-02],
                            [ 6.24143459e-01,  6.00000000e+01, -3.27099727e-02],
                            [ 6.24143459e-01,  1.00000000e+02, -3.27099727e-02],
                            [ 1.12345823e+00,  1.00000000e+02, -5.88779508e-02]]])  
        
        force_good = np.array([[ -7.25540491,   0.        , 596.35460618],
                                [ -7.61281181,   0.        , 601.02845598],
                                [ -7.65793281,   0.        , 601.61167433],
                                [ -7.61281181,   0.        , 601.02845598],
                                [ -7.25540491,   0.        , 596.35460618],
                                [  7.82150574,  -0.        , 198.21738467],
                                [  7.86339581,  -0.        , 199.79346364],
                                [  7.8685363 ,  -0.        , 199.98850418],
                                [  7.86339581,  -0.        , 199.79346364],
                                [  7.82150574,  -0.        , 198.21738467]])  
                          

        force, _, _ = calc_force_wrapper(V_app_infw, gamma_magnitude, self.rho, cp, rings, ns, normals, span_vectors, trailing_edge_info, leading_edge_info, self.gamma_orientation)
        np.testing.assert_almost_equal(force, force_good, decimal=3)

    
    def test_calc_pressure(self):
        normals = np.array([[ 0., -0.,  1.]])
        areas = np.array([100.0]) 
        force = np.array([[3., 2., 1.]])
        press = calc_pressure(force, normals, areas)
        np.testing.assert_almost_equal(press, 0.01)
