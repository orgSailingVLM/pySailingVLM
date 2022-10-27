from dataclasses import dataclass, field
import numpy as np
from sailingVLM.Inlet.InletConditions import InletConditionsNew
from sailingVLM.NewApproach.vlm_logic import \
    get_panels_area, \
    calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points, \
    get_influence_coefficients_spanwise, \
    solve_eq, calc_induced_velocity,  \
    is_no_flux_BC_satisfied, \
    calc_force_wrapper_new, \
    calc_pressure_new_approach, \
    get_vlm_CL_CD_free_wing, \
    create_panels, \
    get_influence_coefficients_spanwise_jib_version, \
    calc_force_wrapper_new_jib_version
    

from sailingVLM.Solver.coeff_formulas import get_CL_CD_free_wing
from typing import List

from sailingVLM.Solver.vlm_solver import calculate_app_fs
# sprawdzic typy tablic !!!!
@dataclass
class Vlm:
    
    chord : float = 1.              # chord length
    half_wing_span : int = 100
    AoA_deg : float =  3.0   # Angle of attack [deg]

    M: int = 5
    N: int = 10
    rho : float = 1.225
    gamma_orientation : float = 1.0
    V : np.array = 1 * np.array([10.0, 0.0, 0.0])
   
    # post init atributes
    areas : np.ndarray = field(init=False)
    normals : np.ndarray = field(init=False)
    collocation_points : np.ndarray = field(init=False)
    center_of_pressure : np.ndarray = field(init=False)
    rings : np.ndarray = field(init=False)
    span_vectors : np.ndarray = field(init=False)
    coefs : np.ndarray = field(init=False)
    RHS : np.ndarray = field(init=False)
    wind_coefs : np.ndarray = field(init=False)
    trailing_rings : np.ndarray = field(init=False)
    big_gamma : np.ndarray = field(init=False)
    F : np.ndarray = field(init=False)
    pressure : np.ndarray = field(init=False)
    panels : np.ndarray = field(init=False)
    
    AR : float =  field(init=False)
    S : float  = field(init=False)
    CL_expected : float = field(init=False)
    CD_ind_expected : float = field(init=False)
    CL_vlm : float = field(init=False)
    CD_vlm : float = field(init=False)
    
    def __post_init__(self):
        
        self.panels = create_panels(self.half_wing_span, self.chord, self.AoA_deg, self.M, self.N)

        V_app_infw = np.array([self.V for i in range(self.M * self.N)])

        self.areas = get_panels_area(self.panels, self.N, self.M) 
        self.normals, self.collocation_points, self.center_of_pressure, self.rings, self.span_vectors = calculate_normals_collocations_cps_rings_spans(self.panels, self.gamma_orientation)
        self.coefs, self.RHS, self.wind_coefs = get_influence_coefficients_spanwise(self.collocation_points, self.rings, self.normals, self.M, self.N, V_app_infw)
        self.big_gamma = solve_eq(self.coefs, self.RHS)
        
        V_induced_at_ctrl_p = calc_induced_velocity(self.wind_coefs, self.big_gamma)
        V_app_fw_at_ctrl_p = V_app_infw + V_induced_at_ctrl_p
        assert is_no_flux_BC_satisfied(V_app_fw_at_ctrl_p, self.panels, self.areas, self.normals)
        self.F = calc_force_wrapper_new(V_app_infw, self.big_gamma, self.panels, self.rho, self.center_of_pressure, self.rings, self.M, self.N, self.normals, self.span_vectors)
        self.pressure = calc_pressure_new_approach(self.F, self.normals, self.areas, self.N, self.M)
        
        self.AR = 2 * self.half_wing_span / self.chord
        self.S = 2 * self.half_wing_span * self.chord
        self.CL_expected, self.CD_ind_expected = get_CL_CD_free_wing(self.AR, self.AoA_deg)
        
        self.CL_vlm, self.CD_vlm = get_vlm_CL_CD_free_wing(self.F, self.V, self.rho, self.S)
        
        

# sprawdzic typy !!!!
@dataclass
class NewVlm:
    

    panels: np.ndarray
    n_chordwise : int
    n_spanwise :int 
        
    rho : float
    wind : np.ndarray
    #V_app_infs : np.ndarray
    sails : List[np.ndarray]
    trailing_edge_info : np.ndarray
    leading_edge_info : np.ndarray  
    gamma_orientation : float = 1.0
    
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
    inlet_conditions: InletConditionsNew = field(init=False)
    force : np.ndarray = field(init=False)
    pressure : np.ndarray = field(init=False)
    V_app_fs_at_cp : np.ndarray = field(init=False)
    V_induced_at_cp : np.ndarray = field(init=False)
    
    def __post_init__(self):
        
        # M = wzdluz rozpoetosci skrzydel, spanwise
        # N = chordwise, linia laczaca leading i trailing
        self.areas = get_panels_area(self.panels) 
        self.normals, self.collocation_points, self.center_of_pressure, self.rings, self.span_vectors, self.leading_mid_points, self.trailing_mid_points = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(self.panels, self.gamma_orientation)
        
        self.inlet_conditions = InletConditionsNew(self.wind, self.rho, self.center_of_pressure)
        
        self.coefs, self.RHS, self.wind_coefs = get_influence_coefficients_spanwise_jib_version( self.collocation_points, self.rings, self.normals, self.inlet_conditions.V_app_infs, self.sails, self.trailing_edge_info, self.gamma_orientation)
        self.gamma_magnitude = solve_eq( self.coefs,  self.RHS)

        self.V_induced_at_ctrl,  self.V_app_fs_at_ctrl_p = calculate_app_fs(self.inlet_conditions.V_app_infs,  self.wind_coefs,  self.gamma_magnitude)

        assert is_no_flux_BC_satisfied(self.V_app_fs_at_ctrl_p, self.panels, self.areas, self.normals)
        
        self.force, self.V_app_fs_at_cp, self.V_induced_at_cp = calc_force_wrapper_new_jib_version(self.inlet_conditions.V_app_infs, self.gamma_magnitude, self.rho, self.center_of_pressure, self.rings, self.n_spanwise, self.n_chordwise, self.normals, self.span_vectors, self.sails, self.trailing_edge_info, self.leading_edge_info, self.gamma_orientation)
    

    
        self.pressure = calc_pressure_new_approach(self.force, self.normals, self.areas, self.n_spanwise, self.n_chordwise)
        
    
        

        
