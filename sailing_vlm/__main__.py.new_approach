import pstats
import numpy as np
import cProfile
import time
import sys, os
from pstats import SortKey
from contextlib import redirect_stdout
### GEOMETRY DEFINITION ###


def new_approach(chord : float, half_wing_span : float, AoA_deg : float, V : np.array, rho : float, ns : int, nc : int, gamma_orientation : float, print_text : bool):

    from sailingVLM.NewApproach.vlm import Vlm

    np.set_printoptions(precision=10, suppress=True)


    my_vlm = Vlm(chord=chord, half_wing_span=half_wing_span, AoA_deg=AoA_deg, M=ns, N=nc, rho=rho, gamma_orientation=1.0, V=V)

    if print_text:
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

def old_approach(chord : float, half_wing_span : float, AoA_deg : float, V : np.array, rho : float, ns : int, nc : int, gamma_orientation : float, print_text):

    from sailingVLM.Solver.vlm_solver import calc_circulation
    from sailingVLM.Solver.mesher import make_panels_from_le_te_points
    from sailingVLM.Rotations.geometry_calc import rotation_matrix
    from sailingVLM.Solver.coeff_formulas import get_CL_CD_free_wing
    from sailingVLM.Solver.forces import calc_force_wrapper, calc_pressure
    from sailingVLM.Solver.forces import calc_force_wrapper_new
    from sailingVLM.Solver.vlm_solver import is_no_flux_BC_satisfied, calc_induced_velocity


    np.set_printoptions(precision=3, suppress=True)

    
    # Points defining wing (x,y,z) #
    le_NW = np.array([0., half_wing_span, 0.])      # leading edge North - West coordinate
    le_SW = np.array([0., -half_wing_span, 0.])     # leading edge South - West coordinate

    te_NE = np.array([chord, half_wing_span, 0.])   # trailing edge North - East coordinate
    te_SE = np.array([chord, -half_wing_span, 0.])  # trailing edge South - East coordinate

   
    Ry = rotation_matrix([0, 1, 0], np.deg2rad(AoA_deg))
    # we are going to rotate the geometry

    ### MESH DENSITY ###
    
    panels, mesh, _ = make_panels_from_le_te_points(
        [np.dot(Ry, le_SW),
        np.dot(Ry, te_SE),
        np.dot(Ry, le_NW),
        np.dot(Ry, te_NE)],
        [nc, ns],
        gamma_orientation=gamma_orientation)

    rows, cols = panels.shape
    N = rows * cols

    
    ### FLIGHT CONDITIONS ###
    V_app_infw = np.array([V for i in range(N)])
   
    ### CALCULATIONS ###
    gamma_magnitude, v_ind_coeff, A = calc_circulation(V_app_infw, panels)
    V_induced_at_ctrl_p = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
    V_app_fw_at_ctrl_p = V_app_infw + V_induced_at_ctrl_p
    assert is_no_flux_BC_satisfied(V_app_fw_at_ctrl_p, panels)

    Fold = calc_force_wrapper(V_app_infw, gamma_magnitude, panels, rho=rho)

    F = calc_force_wrapper_new(V_app_infw, gamma_magnitude, panels, rho)
    F = F.reshape(N, 3)

    p = calc_pressure(F, panels)
    if print_text:
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

    if print_text:
        print(f"\nAspect Ratio {AR}")
        print(f"CL_expected {CL_expected:.6f} \t CD_ind_expected {CD_ind_expected:.6f}")
        print(f"CL_vlm      {CL_vlm:.6f}  \t CD_vlm          {CD_vlm:.6f}")

        print(f"\n\ntotal_F {str(total_F)}")
        print(f"total_Fold {str(np.sum(Fold, axis=0))}")
        print("=== END ===")

def main(print_text : bool):
    with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            chord = 1.              # chord length
            half_wing_span = 100.
            AoA_deg = 3.0
            V = 1*np.array([10.0, 0.0, 0.0])
            rho = 1.225  # fluid density [kg/m3]

            ns = 20    # number of panels (spanwise)
            nc = 20    # number of panels (chordwise)
            gamma_orientation = 1.0
            #old_approach(chord, half_wing_span, AoA_deg, V, rho, ns, nc, gamma_orientation, print_text)
            new_approach(chord, half_wing_span, AoA_deg, V, rho, ns, nc, gamma_orientation, print_text)
        
if __name__ == "__main__":
    
    
    start = time.time()
    # profile code if main takes no arguments
    #cProfile.run("main()", "output.dat")\# profile main with argumants
    #cProfile.runctx('main(print_text)', {'print_text': True, 'main' : main}, {}, "output.dat")
    
    cProfile.runctx('main(print_text)', {'print_text': True, 'main' : main}, {}, "output.dat")
    #cProfile.runctx('old_approach(chord, half_wing_span, AoA_deg, V, rho, ns, nc, gamma_orientation)', {'chord': chord, 'half_wing_span' : half_wing_span, 'AoA_deg': AoA_deg,  'V' : V, 'rho' : rho, 'ns' : ns, 'nc' : nc, 'gamma_orientation' : gamma_orientation}, {}, filename='output.dat')
    end = time.time()
    print("Elapsed = %s" % (end - start))

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()
        
    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()
        
            