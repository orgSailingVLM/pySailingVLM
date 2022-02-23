from Solver.Panel import Panel
from Solver.vortices import v_induced_by_horseshoe_vortex

from Solver.vortices import \
    v_induced_by_semi_infinite_vortex_line, \
    v_induced_by_finite_vortex_line, \
    v_induced_by_horseshoe_vortex, \
    normalize

class TrailingEdgePanel(Panel):
    def __init__(self, p1, p2, p3, p4, gamma_orientation=1):
        super().__init__(p1, p2, p3, p4, gamma_orientation=1)

    """
    def get_induced_velocity(self, ctr_p, V_app_infw, gamma=1):
        print("Tutaj")
        [A, B, C, D] = self.get_vortex_ring_position()
        v = v_induced_by_horseshoe_vortex(ctr_p, B, C, V_app_infw, self.gamma_orientation)
        return v
        
    """

    def get_induced_velocity(self, ctr_p, V_app_infw):
        [A, B, C, D] = self.get_vortex_ring_position()
        v = v_induced_by_horseshoe_vortex(ctr_p, B, C, V_app_infw, self.gamma_orientation)
        return v
