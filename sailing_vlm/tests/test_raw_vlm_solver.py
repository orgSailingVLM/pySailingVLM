import numpy as np
from unittest import TestCase

from sailing_vlm.solver.panels import make_panels_from_le_te_points, get_panels_area
from sailing_vlm.solver.coefs import calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points, \
    get_influence_coefficients_spanwise

class TestRawSolver(TestCase):
    def setUp(self):
        #GEOMETRY DEFINITION #
        #Parameters
        c_root = 2.0  # root chord length
        c_tip = 2.0  # tip chord length
        half_wing_span = 10.  # wing span length

        #Points defining wing
        le_root_coord = np.array([0., -half_wing_span, 0.])
        le_tip_coord = np.array([0., half_wing_span, 0])

        te_root_coord = np.array([c_root, -half_wing_span, 0])
        te_tip_coord = np.array([c_tip, half_wing_span, 0])

        #MESH DENSITY
        ns = 3  # number of panels (spanwise)
        nc = 1  # number of panels (chordwise)

        self.panels, self.trailing_edge_info, self.leading_edge_info = make_panels_from_le_te_points([le_root_coord, te_root_coord, le_tip_coord, te_tip_coord], [nc, ns])

        self.N = ns * nc
        self.gamma_orientation = 1
        self.areas = get_panels_area(self.panels) 
        self.normals, self.collocation_points, _, self.rings, _, _, _ = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(self.panels, self.gamma_orientation)


    def test_matrix_symmetry(self):
        import random

        for i in range(50):
            V = [random.uniform(-10, 10), 0, random.uniform(-10, 10)]
            V_free_stream = np.array([V for i in range(self.N)])

            coefs, _, _ = get_influence_coefficients_spanwise(self.collocation_points, self.rings, self.normals, V_free_stream, self.trailing_edge_info, self.gamma_orientation)
            is_mat_symmeric = np.allclose(coefs, coefs.T, atol=1e-8)
            assert is_mat_symmeric
