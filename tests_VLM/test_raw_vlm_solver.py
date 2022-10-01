import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase

from Solver.mesher import make_panels_from_le_te_points

from Solver.vlm_solver import \
    assembly_sys_of_eq, \
    calc_circulation, \
    is_no_flux_BC_satisfied, \
    calc_induced_velocity


class TestVLM_Solver(TestCase):
    def setUp(self):
        # GEOMETRY DEFINITION #
        # Parameters
        c_root = 2.0  # root chord length
        c_tip = 2.0  # tip chord length
        half_wing_span = 10.  # wing span length

        # Points defining wing
        le_root_coord = np.array([0., -half_wing_span, 0.])
        le_tip_coord = np.array([0., half_wing_span, 0])

        te_root_coord = np.array([c_root, -half_wing_span, 0])
        te_tip_coord = np.array([c_tip, half_wing_span, 0])

        # MESH DENSITY
        ns = 3  # number of panels (spanwise)
        nc = 1  # number of panels (chordwise)

        self.panels, _ = make_panels_from_le_te_points(
            [le_root_coord, te_root_coord, le_tip_coord, te_tip_coord],
            [nc, ns], gamma_orientation=1)

        self.N = ns * nc

    def test_matrix_symmetry(self):
        import random

        for i in range(50):
            V = [random.uniform(-10, 10), 0, random.uniform(-10, 10)]
            V_free_stream = np.array([V for i in range(self.N)])

            A, RHS, _ = assembly_sys_of_eq(V_free_stream, self.panels)
            is_mat_symmeric = np.allclose(A, A.T, atol=1e-8)
            assert is_mat_symmeric

    def test_calc_circulation(self):
        V = [10, 0, -1]  # [m/s] wind speed
        V_free_stream = np.array([V for i in range(self.N)], dtype=np.float64)

        gamma_magnitude, v_ind_coeff = calc_circulation(V_free_stream, self.panels)

        gamma_expected = [-5.26437093, -5.61425005, -5.26437093]
        assert_almost_equal(gamma_magnitude, gamma_expected)

        V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
        V_app_fs = V_free_stream + V_induced
        assert is_no_flux_BC_satisfied(V_app_fs, self.panels)

        with self.assertRaises(ValueError) as context:
            V_broken = 1e10 * V_app_fs
            is_no_flux_BC_satisfied(V_broken, self.panels)

        self.assertTrue("Solution error, there shall be no flow through panel!" in context.exception.args[0])
