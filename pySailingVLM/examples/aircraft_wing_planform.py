import timeit
import numpy as np


from pySailingVLM.rotations.geometry_calc import rotation_matrix
from pySailingVLM.solver.panels import make_panels_from_le_te_points, get_panels_area
from pySailingVLM.solver.coefs import get_CL_CD_free_wing, get_vlm_CL_CD_free_wing
from pySailingVLM.solver.coefs import calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points, \
                                solve_eq, calculate_RHS, calc_velocity_coefs

from pySailingVLM.solver.velocity import calculate_app_fs 
from pySailingVLM.solver.forces import is_no_flux_BC_satisfied, calc_force_wrapper

### GEOMETRY DEFINITION ###

"""
    This example shows how to use the pyVLM class in order
    to generate the wing planform.

    After defining the flight conditions (airspeed and AOA),
    the geometry will be characterised using the following
    nomenclature:

    Y  ^    le_NW +--+ te_NE
       |         /    \
       |        /      \
       |       /        \
       +------/----------\---------------->
       |     /            \               X
       |    /              \
     le_SW +-----------------+ te_SE
     
 
"""
start = timeit.default_timer()
np.set_printoptions(precision=3, suppress=True)

# PARAMETERS
gamma_orientation = 1
### WING DEFINITION ###
#Parameters #
chord = 1.              # chord length
half_wing_span = 100.    # wing span length

# Points defining wing (x,y,z) #
le_NW = np.array([0., half_wing_span, 0.])      # leading edge North - West coordinate
le_SW = np.array([0., -half_wing_span, 0.])     # leading edge South - West coordinate

te_NE = np.array([chord, half_wing_span, 0.])   # trailing edge North - East coordinate
te_SE = np.array([chord, -half_wing_span, 0.])  # trailing edge South - East coordinate

AoA_deg = 3.0   # Angle of attack [deg]
Ry = rotation_matrix([0, 1, 0], np.deg2rad(AoA_deg))
# we are going to rotate the geometry

### MESH DENSITY ###

ns = 5   # number of panels (spanwise)
nc = 5   # number of panels (chordwise)

panels, trailing_edge_info, leading_edge_info = make_panels_from_le_te_points(
    [np.dot(Ry, le_SW),
    np.dot(Ry, te_SE),
    np.dot(Ry, le_NW),
    np.dot(Ry, te_NE)],
    [nc, ns])

N = panels.shape[0]

### FLIGHT CONDITIONS ###
V = 1*np.array([10.0, 0.0, 0.0])
V_app_infw = np.array([V for i in range(N)])
rho = 1.225  # fluid density [kg/m3]

AR = 2 * half_wing_span / chord
S = 2 * half_wing_span * chord
CL_expected, CD_expected = get_CL_CD_free_wing(AR, AoA_deg)

################## CALCULATIONS ##################

areas = get_panels_area(panels) 
normals, collocation_points, center_of_pressure, rings, span_vectors, _, _ = calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(panels, gamma_orientation)

coefs, v_ind_coeff = calc_velocity_coefs(V_app_infw, collocation_points, rings, normals, trailing_edge_info, gamma_orientation)
RHS = calculate_RHS(V_app_infw, normals)
gamma_magnitude = solve_eq(coefs, RHS)

_,  V_app_fs_at_ctrl_p = calculate_app_fs(V_app_infw,  v_ind_coeff,  gamma_magnitude)
assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, panels, areas, normals)

force, _, _ = calc_force_wrapper(V_app_infw, gamma_magnitude, rho, center_of_pressure, rings, ns, normals, span_vectors, trailing_edge_info, leading_edge_info, gamma_orientation)

CL_vlm, CD_vlm = get_vlm_CL_CD_free_wing(force, V, rho, S)

print("gamma_magnitude: \n")
print(gamma_magnitude)
print("DONE")


print(f"\nAspect Ratio {AR}")
print(f"CL_expected {CL_expected:.6f} \t CD_ind_expected {CD_expected:.6f}")
print(f"CL_vlm      {CL_vlm:.6f}  \t CD_vlm          {CD_vlm:.6f}")

print(f"\n\ntotal_F {str(np.sum(force, axis=0))}")
print("=== END ===")

print(f"\nCPU time: {float(timeit.default_timer() - start):.2f} [s]")
