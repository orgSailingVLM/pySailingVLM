import numpy as np
from unittest import TestCase


class TestValidation(TestCase):
    
#     >>> import sailing_vlm.runner.aircraft as ac
# >>> import numpy as np
# >>> a = ac.Aircraft(1.0, 5.0, 10.0, 32, 8, np.array([1.0, .0, .0]), 1.225, -1.0)
# >>> a.get_Cxyz()
# (0.023472780216173314, 0.8546846987984326, 0.0, array([0.14377078, 5.23494378, 0.        ]), array([1., 0., 0.]), 10.0, 6.125)
# ```
# SAIL
# 0.023472780216173342 0.8546846987984335 6.996235308182204e-34 [1.43770779e-01 5.23494378e+00 4.28519413e-33] [1. 0. 0.] 10.0 6.125
#     
    def test_flat_plate(self):
        chord_length = 1.0
        half_span_length = 5.0
        n_spanwise = 32
        n_chordwise = 8
        V = np.array([1.0, 0., 0.])
        rho = 1.225
        gamma_orientation = -1.0