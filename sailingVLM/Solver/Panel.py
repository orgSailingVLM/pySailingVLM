import numpy as np
import warnings

from sailingVLM.Solver.vortices import \
    v_induced_by_semi_infinite_vortex_line, \
    v_induced_by_finite_vortex_line, \
    v_induced_by_horseshoe_vortex, \
    normalize


class Panel(object):
    """
           y ^
             |              Each panel is defined by the (x, y) coordinates
        P3-C-|-D-P4         of four points - namely P1, P2, P3 and P4 -
         | | | |  |         ordered clockwise. Points defining the horseshoe
         | | +-P--|--->     - A, B, C and D - are named clockwise as well.
         | |   |  |   x
        P2-B---A-P1

    Parameters
    ----------
    P1, P2, P3, P4 : array_like
                     Corner points in a 3D euclidean space
    """
    panel_counter = 0
    _are_no_coplanar_panels_reported = False

    def __init__(self, p1, p2, p3, p4, gamma_orientation=1):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.gamma_orientation = gamma_orientation  # it can be turning clock or counter-clock wise
        # 1 is for a horizontal wing (glider), when air is flowing from below
        # -1 is for a vertical sail, when the wind is going from your back as you look towards the sail
        self.counter = Panel.panel_counter
        Panel.panel_counter += 1

        self.pressure = None
        self.force_xyz = None
        self.V_app_fs_at_cp = None
        self.V_induced_at_cp = None

        if not self._are_points_coplanar() and not Panel._are_no_coplanar_panels_reported:
            print("Panels are not coplanar (twisted).")
            Panel._are_no_coplanar_panels_reported = True
            # warnings.warn("Points on Panel are not coplanar!")
            # raise ValueError("Points on Panel are not coplanar!")

    def calc_pressure(self):
        area = self.get_panel_area()
        n = self.get_normal_to_panel()
        self.pressure = np.dot(self.force_xyz, n) / area  # todo: fix sign

    def _are_points_coplanar(self):
        # P1P2 = self.p1 - self.p2
        # P3P4 = self.p4 - self.p3
        #
        # vec_perpendicular = np.cross(P1P2, P3P4)
        # v1 = np.dot(vec_perpendicular, P1P2)
        # v2 = np.dot(vec_perpendicular, P3P4)

        # http://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
        # A plane is defined by the equation:
        # ax+by+cz=d
        # These two vectors are in the plane
        v1 = self.p3 - self.p1
        v2 = self.p2 - self.p1

        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp

        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, self.p3)

        d2 = a*self.p4[0] + b*self.p4[1] + c*self.p4[2]
        if abs(d-d2) > 1e-12:
            return False

        # if np.linalg.norm(vec_perpendicular) > 1e-12:
        #     raise ValueError("Points on Panel are not colinear!")

        return True

    def get_normal_to_panel(self):
        # definition
        # for the wing in x-y plane the normal is assumed to be z-axis-positive,
        # and circulation is assumed to be z-axis-negative

        # formula from Katz and Plotkin,
        #  Fig 12.11 p 343, Chapter 12.3 Lifting-Surface Solution by Vortex Ring Elements
        p2_p4 = self.p4 - self.p2
        p1_p3 = self.p3 - self.p1
        n = np.cross(p2_p4, p1_p3)
        # n = np.cross(p1_p3, p2_p4)
        n = normalize(n)
        return n

    def get_panel_area(self):
        p = [self.p1, self.p2, self.p3, self.p4]

        path = []
        for i in range(len(p) - 1):
            step = p[i + 1] - p[i]
            path.append(step)

        # path.append(p[3] - p[0])
        # path.append(p[0] - p[1])

        area = 0
        for i in range(len(path) - 1):
            s = np.cross(path[i], path[i + 1])
            s = np.linalg.norm(s)
            area += 0.5 * s

        return area

    def get_span_vector(self):
        [A, B, C, D] = self.get_vortex_ring_position()
        bc = C - B
        bc *= self.gamma_orientation
        return np.array(bc)

    def get_panel_span_at_cp(self):
        # this is length of vortex line going through centre of pressure
        [_, B, C, _] = self.get_vortex_ring_position()
        BC = C-B
        BC_length = np.linalg.norm(BC)
        return BC_length

    def get_points(self):
        return self.p1, self.p2, self.p3, self.p4

    def get_leading_edge_mid_point(self):
        return (self.p2 + self.p3)/2.

    def get_trailing_edge_mid_points(self):
        return (self.p4 + self.p1)/2.

    def get_ctr_point_position(self):
        """
         For a given panel defined by points P1, P2, P3 and P4
         returns the position of the control point P.

                  ^
                 y|                Points defining the panel
                  |                are named clockwise.
          P3------|------P4
           |      |      |
           |      |      |
           |      +---P----------->
           |             |      x
           |             |
          P2-------------P1

         Parameters
         ----------
         P1, P2, P3, P4 : array_like
                          Points that define the panel

         Returns
         -------
             P - control point where the boundary condition V*n = 0
                 is applied according to the Vortice Lattice Method.
                 It shall be located at 3/4 of the chord.
         """
        # p2_p1 = self.p1 - self.p2
        # p1_p4 = self.p4 - self.p1
        # ctr_p = self.p2 + p2_p1 * (3. / 4.) + p1_p4 / 2.

        #new
        le_mid_point = self.get_leading_edge_mid_point()
        te_mid_point = self.get_trailing_edge_mid_points()
        tl = te_mid_point - le_mid_point
        ctr_p = le_mid_point + (3. / 4.) * tl
        return ctr_p

    @property
    def cp_position(self):
        """
         For a given panel defined by points P1, P2, P3 and P4
         returns the position of the centre of pressure

                  ^
                 y|                Points defining the panel
                  |                are named clockwise.
                  |
          P3--:...|......P4....................... lifting line
           |  :   |      |
           |  :   |      |
           |  CP  +--------------->
           |  :          |      x
           |  :          |
          P2--:..........P1....................... lifting line

         Parameters
         ----------
         P1, P2, P3, P4 : array_like
                          Points that define the panel

         Returns
         -------
         results : dict
             CP - centre of pressure, when calculating CL & CD it assumed that the force is attached to this point.
             The induced wind is calculated at CP, and then U_inf + U_ind is used to find the force.
             The CP shall be located at 1/4 of the chord.
         """
        # p2_p1 = self.p1 - self.p2
        # p1_p4 = self.p4 - self.p1
        # cp = self.p2 + p2_p1 * (1. / 4.) + p1_p4 / 2.

        #new
        le_mid_point = self.get_leading_edge_mid_point()
        te_mid_point = self.get_trailing_edge_mid_points()
        tl = te_mid_point - le_mid_point
        cp = le_mid_point + (1. / 4.) * tl
        return cp

    def get_vortex_ring_position(self):
        """
        For a given panel defined by points P1, P2, P3 and P4
        returns the position of the horseshoe vortex defined
        by points A, B and its control point P.

                  ^
                 y|                Points defining the panel
                  |                are named clockwise.
         P3--C----|---P4---D
          |  |    |    |   |
          |  |    |    |   |
          |  |    +----|---------->
          |  |         |   |      x
          |  |         |   |
         P2--B---------P1--A

        Parameters
        ----------
        P1, P2, P3, P4 : array_like
                         Points that define the panel

        Returns
        -------
        results : dict
            A, B, C, D - points that define the vortex ring
        """

        p2_p1 = self.p1 - self.p2
        p3_p4 = self.p4 - self.p3

        A = self.p1 + p2_p1 / 4.
        B = self.p2 + p2_p1 / 4.
        C = self.p3 + p3_p4 / 4.
        D = self.p4 + p3_p4 / 4.

        return [A, B, C, D]

    def get_induced_velocity(self, ctr_p, V_app_infw):
        [A, B, C, D] = self.get_vortex_ring_position()

        v_AB = v_induced_by_finite_vortex_line(ctr_p, A, B, self.gamma_orientation)
        v_BC = v_induced_by_finite_vortex_line(ctr_p, B, C, self.gamma_orientation)
        v_CD = v_induced_by_finite_vortex_line(ctr_p, C, D, self.gamma_orientation)
        v_DA = v_induced_by_finite_vortex_line(ctr_p, D, A, self.gamma_orientation)

        v = v_AB + v_BC + v_CD + v_DA
        return v

