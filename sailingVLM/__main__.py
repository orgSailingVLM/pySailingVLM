import numpy as np
from sailingVLM.NewApproach.vlm import Vlm

### GEOMETRY DEFINITION ###


np.set_printoptions(precision=10, suppress=True)

### WING DEFINITION ###
#Parameters #
chord = 1.              # chord length
half_wing_span = 100.    # wing span length
AoA_deg = 3.0   # Angle of attack [deg]


### FLIGHT CONDITIONS ###
V = 1*np.array([10.0, 0.0, 0.0])
rho = 1.225  # fluid density [kg/m3]

ns = 10    # number of panels (spanwise)
nc = 5     # number of panels (chordwise)

my_vlm = Vlm(chord=chord, half_wing_span=half_wing_span, AoA_deg=AoA_deg, M=ns, N=nc, rho=rho, gamma_orientation=1.0, V=V)

print("gamma_magnitude: \n")
print(my_vlm.big_gamma)
print("DONE")

### compare vlm with book formulas ###
# reference values - to compare with book formulas
print(f"\nAspect Ratio {my_vlm.AR}")
print(f"CL_expected {my_vlm.CL_expected:.6f} \t CD_ind_expected {my_vlm.CD_ind_expected:.6f}")
print(f"CL_vlm      {my_vlm.CL_vlm:.6f}  \t CD_vlm          {my_vlm.CD_vlm:.6f}")
print(f"\n\ntotal_F {str(np.sum(my_vlm.F, axis=0))}")
# print(f"total_Fold {str(np.sum(Fold, axis=0))}")
print("=== END ===")
