from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sailing_vlm.inlet.inlet_conditions import InletConditions
from sailing_vlm.solver import coefs
from sailing_vlm.solver import forces
from sailing_vlm.solver import velocity
from sailing_vlm.solver import panels
from typing import List
from typing import ClassVar
from sailing_vlm.solver.panels_plotter import display_panels_or_rings

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

# sprawdzic typy !!!!
@dataclass
class Vlm:


    panels: np.ndarray
    n_chordwise : int
    n_spanwise :int

    rho : float
    wind : np.ndarray
    trailing_edge_info : np.ndarray
    leading_edge_info : np.ndarray


    # post init atributes
    areas : np.ndarray = field(init=False)
    normals : np.ndarray = field(init=False)
    leading_mid_points : np.ndarray = field(init=False)
    trailing_mid_points : np.ndarray = field(init=False)
    ctr_p : np.ndarray = field(init=False) # control point in 3/4 of chord -> other name: collocation point 
    cp : np.ndarray = field(init=False) # 1/4 length of chord -> other name: center of pressure
    rings : np.ndarray = field(init=False)
    span_vectors : np.ndarray = field(init=False)
    coeff : np.ndarray = field(init=False)
    RHS : np.ndarray = field(init=False)
    v_ind_coeff : np.ndarray = field(init=False)
    gamma_magnitude : np.ndarray = field(init=False)
    inlet_conditions: InletConditions = field(init=False)
    force : np.ndarray = field(init=False)
    pressure : np.ndarray = field(init=False)
    V_app_fs_at_cp : np.ndarray = field(init=False)
    V_induced_at_cp : np.ndarray = field(init=False)

    # as class var
    gamma_orientation : float = -1.0

    def __post_init__(self):

        # M = wzdluz rozpoetosci skrzydel, spanwise
        # N = chordwise, linia laczaca leading i trailing
        self.areas = panels.get_panels_area(self.panels)
        #self.normals, self.ctr_p, self.cp, self.rings, self.span_vectors, self.leading_mid_points, self.trailing_mid_points = coefs.calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(self.panels, self.gamma_orientation)
        self.normals, self.ctr_p, self.cp, self.rings, self.span_vectors, self.leading_mid_points, self.trailing_mid_points = coefs.calculate_vlm_variables(self.panels, self.trailing_edge_info, self.gamma_orientation, self.n_chordwise, self.n_spanwise)
        self.inlet_conditions = InletConditions(self.wind, self.rho, self.cp)
        self.coeff, self.v_ind_coeff = coefs.calc_velocity_coefs(self.inlet_conditions.V_app_infs, self.ctr_p, self.rings, self.normals, self.trailing_edge_info, self.gamma_orientation)
        self.RHS = coefs.calculate_RHS(self.inlet_conditions.V_app_infs, self.normals)
        self.gamma_magnitude = coefs.solve_eq(self.coeff, self.RHS)
        self.V_induced_at_ctrl,  self.V_app_fs_at_ctrl_p = velocity.calculate_app_fs(self.inlet_conditions.V_app_infs,  self.v_ind_coeff,  self.gamma_magnitude)

        # boundary condition calculated in collocation points (control points)
        assert forces.is_no_flux_BC_satisfied(self.V_app_fs_at_ctrl_p, self.panels, self.areas, self.normals)
        self.force, self.V_app_fs_at_cp, self.V_induced_at_cp = forces.calc_force_wrapper(self.inlet_conditions.V_app_infs, self.gamma_magnitude, self.rho, self.cp, self.rings, self.n_spanwise, self.normals, self.span_vectors, self.trailing_edge_info, self.leading_edge_info, self.gamma_orientation)
        self.pressure = forces.calc_pressure(self.force, self.normals, self.areas)

        try:
            pressure_signs = np.sign(self.pressure).astype(int)
            result = np.all(pressure_signs == pressure_signs[0])
            if not result:
                raise ValueError('ERROR!: Pressure signs are not the same, check input data!\nKeep in mind that this program generates camber in one direction only.')
        except ValueError as err:
            print(err)
            sys.exit()

        #display_panels_or_rings(self.rings, self.pressure, self.leading_mid_points, self.trailing_mid_points)
        # comment this section if debugging camber
        #plt.show()

