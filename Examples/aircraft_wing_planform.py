import numpy as np
import timeit

from Solver.vlm_solver import calc_circulation
from Solver.mesher import make_panels_from_le_te_points
from Rotations.geometry_calc import rotation_matrix
from Solver.coeff_formulas import get_CL_CD_free_wing
from Solver.forces import calc_forces_on_panels_VLM_xyz, get_stuff_from_panels
from Solver.vlm_solver import is_no_flux_BC_satisfied, calc_induced_velocity

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
ns = 15    # number of panels (spanwise)
nc = 15   # number of panels (chordwise)

panels, mesh = make_panels_from_le_te_points(
    [np.dot(Ry, le_SW),
     np.dot(Ry, te_SE),
     np.dot(Ry, le_NW),
     np.dot(Ry, te_NE)],
    [nc, ns],
    gamma_orientation=1)

rows, cols = panels.shape
N = rows * cols

### FLIGHT CONDITIONS ###
V = 1*np.array([10.0, 0.0, 0.0])
V_app_infw = np.array([V for i in range(N)])
rho = 1.225  # fluid density [kg/m3]

### CALCULATIONS ###
gamma_magnitude, v_ind_coeff_at_ctr_p = calc_circulation(V_app_infw, panels)
V_induced_at_ctrl_p = calc_induced_velocity(v_ind_coeff_at_ctr_p, gamma_magnitude)
V_app_fw_at_ctrl_p = V_app_infw + V_induced_at_ctrl_p
assert is_no_flux_BC_satisfied(V_app_fw_at_ctrl_p, panels)


calc_forces_on_panels_VLM_xyz(V_app_infw, gamma_magnitude, panels, rho)
F = get_stuff_from_panels(panels, "force_xyz", (panels.shape[0], panels.shape[1], 3))
F = F.reshape(N, 3)

map(lambda x: x.calc_pressure(), panels.flatten())

print("gamma_magnitude: \n")
print(gamma_magnitude)
print("DONE")

### compare vlm with book formulas ###
# reference values - to compare with book formulas
AR = 2 * half_wing_span / chord
S = 2 * half_wing_span * chord
CL_expected, CD_ind_expected = get_CL_CD_free_wing(AR, AoA_deg)

total_F = np.sum(F, axis=0)
q = 0.5 * rho * (np.linalg.norm(V) ** 2) * S
CL_vlm = total_F[2] / q
CD_vlm = total_F[0] / q

print(f"\nAspect Ratio {AR}")
print(f"CL_expected {CL_expected:.6f} \t CD_ind_expected {CD_ind_expected:.6f}")
print(f"CL_vlm      {CL_vlm:.6f}  \t CD_vlm          {CD_vlm:.6f}")

print(f"\n\ntotal_F {str(total_F)}")
print("=== END ===")

print(f"\nCPU time: {float(timeit.default_timer() - start):.2f} [s]")


# po dodaniu TrailingEdge wyniki powinny byc o okolo kilkanacsie procent inne
