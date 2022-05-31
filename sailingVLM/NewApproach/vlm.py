from dataclasses import dataclass, field
import numpy as np
from sailingVLM.NewApproach.vlm_logic import \
    get_panels_area, \
    calculate_normals_collocations_cps_rings_spans, \
    get_influence_coefficients_spanwise, \
    solve_eq, calc_induced_velocity,  \
    is_no_flux_BC_satisfied, \
    calc_force_wrapper_new, \
    calc_pressure, \
    get_vlm_CL_CD_free_wing, \
    create_panels

from sailingVLM.Solver.coeff_formulas import get_CL_CD_free_wing

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
    CL : float = field(init=False)
    CD : float = field(init=False)
    
    def __post_init__(self):
        
        self.panels = create_panels(self.half_wing_span, self.chord, self.AoA_deg, self.M, self.N)

        V_app_infw = np.array([self.V for i in range(self.M * self.N)])

        self.areas = get_panels_area(self.panels, self.N, self.M) 
        self.normals, self.collocation_points, self.center_of_pressure, self.rings, self.span_vectors = calculate_normals_collocations_cps_rings_spans(self.panels, self.gamma_orientation)
        self.coefs, self.RHS, self.wind_coefs, self.trailing_rings = get_influence_coefficients_spanwise(self.collocation_points, self.rings, self.normals, self.M, self.N, V_app_infw)
        self.big_gamma = solve_eq(self.coefs, self.RHS)
        
        V_induced_at_ctrl_p = calc_induced_velocity(self.wind_coefs, self.big_gamma)
        V_app_fw_at_ctrl_p = V_app_infw + V_induced_at_ctrl_p
        assert is_no_flux_BC_satisfied(V_app_fw_at_ctrl_p, self.panels, self.areas, self.normals)
        self.F = calc_force_wrapper_new(V_app_infw, self.big_gamma, self.panels, self.rho, self.center_of_pressure, self.rings, self.M, self.N, self.normals, self.span_vectors)
        self.pressure = calc_pressure(self.F, self.normals, self.areas, self.N, self.M)
        
        self.AR = 2 * self.half_wing_span / self.chord
        self.S = 2 * self.half_wing_span * self.chord
        self.CL_expected, self.CD_ind_expected = get_CL_CD_free_wing(self.AR, self.AoA_deg)
        
        self.CL, self.CD = get_vlm_CL_CD_free_wing(self.F, self.V, self.rho, self.S)
        
        


