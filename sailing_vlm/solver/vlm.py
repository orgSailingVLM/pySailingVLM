from dataclasses import dataclass, field
import numpy as np
from sailing_vlm.inlet.inlet_conditions import InletConditions
from sailing_vlm.solver import coefs 
from sailing_vlm.solver import forces 
from sailing_vlm.solver import velocity 
from sailing_vlm.solver import panels 
from typing import List
from typing import ClassVar


# sprawdzic typy !!!!
@dataclass
class Vlm:
    

    panels: np.ndarray
    n_chordwise : int
    n_spanwise :int 
        
    rho : float
    wind : np.ndarray
    sails : List[np.ndarray]
    trailing_edge_info : np.ndarray
    leading_edge_info : np.ndarray  
    
    
    # post init atributes
    areas : np.ndarray = field(init=False)
    normals : np.ndarray = field(init=False)
    leading_mid_points : np.ndarray = field(init=False)
    trailing_mid_points : np.ndarray = field(init=False)
    collocation_points : np.ndarray = field(init=False)
    center_of_pressure : np.ndarray = field(init=False)
    rings : np.ndarray = field(init=False)
    span_vectors : np.ndarray = field(init=False)
    coefs : np.ndarray = field(init=False)
    RHS : np.ndarray = field(init=False)
    wind_coefs : np.ndarray = field(init=False)
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
        self.normals, self.collocation_points, self.center_of_pressure, self.rings, self.span_vectors, self.leading_mid_points, self.trailing_mid_points = coefs.calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(self.panels, self.gamma_orientation)
        
        self.inlet_conditions = InletConditions(self.wind, self.rho, self.center_of_pressure)
        
        self.coefs, self.RHS, self.wind_coefs = coefs.get_influence_coefficients_spanwise( self.collocation_points, self.rings, self.normals, self.inlet_conditions.V_app_infs, self.trailing_edge_info, self.gamma_orientation)
        self.gamma_magnitude = coefs.solve_eq( self.coefs,  self.RHS)

        self.V_induced_at_ctrl,  self.V_app_fs_at_ctrl_p = velocity.calculate_app_fs(self.inlet_conditions.V_app_infs,  self.wind_coefs,  self.gamma_magnitude)

        assert forces.is_no_flux_BC_satisfied(self.V_app_fs_at_ctrl_p, self.panels, self.areas, self.normals)
        
        self.force, self.V_app_fs_at_cp, self.V_induced_at_cp = forces.calc_force_wrapper(self.inlet_conditions.V_app_infs, self.gamma_magnitude, self.rho, self.center_of_pressure, self.rings, self.n_spanwise, self.n_chordwise, self.normals, self.span_vectors, self.sails, self.trailing_edge_info, self.leading_edge_info, self.gamma_orientation)

        self.pressure = forces.calc_pressure(self.force, self.normals, self.areas, self.n_spanwise, self.n_chordwise)
        
