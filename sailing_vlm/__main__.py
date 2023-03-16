import timeit
import shutil


from sailing_vlm.rotations.csys_transformations import CSYS_transformations
from sailing_vlm.yacht_geometry.hull_geometry import HullGeometry


from sailing_vlm.results.save_utils import save_results_to_file
from sailing_vlm.solver.panels_plotter import display_panels_xyz_and_winds


from sailing_vlm.results.inviscid_flow import InviscidFlowResults

#from sailing_vlm.examples.input_data.prostokat import *
from sailing_vlm.examples.input_data.jib_and_main_sail_vlm_case import *
from sailing_vlm.solver.coefs import get_vlm_CL_CD_free_wing, get_vlm_Cxyz
from sailing_vlm.solver.vlm import Vlm

from sailing_vlm.runner.sail import Wind, Sail


import pstats
import numpy as np
import cProfile
import time
import sys, os


def main():
    
    csys_transformations = CSYS_transformations(
        heel_deg, leeway_deg,
        v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

    w = Wind(alpha_true_wind_deg, tws_ref,SOG_yacht, wind_exp_coeff, wind_reference_measurment_height, reference_water_level_for_wind_profile, roughness, wind_profile)
    w_profile = w.profile

    s = Sail(n_spanwise, n_chordwise, csys_transformations, sheer_above_waterline,
             rake_deg, boom_above_sheer, mast_LOA,
             main_sail_luff, main_sail_girths, main_sail_chords, main_sail_centerline_twist_deg, main_sail_camber,main_sail_camber_distance_from_luff,
            foretriangle_base, foretriangle_height, 
            jib_luff, jib_girths, jib_chords, jib_centerline_twist_deg, jib_sail_camber, jib_sail_camber_distance_from_luff,
            sails_def, LLT_twist, interpolation_type)
    sail_set = s.sail_set
    hull = HullGeometry(sheer_above_waterline, foretriangle_base, csys_transformations, center_of_lateral_resistance_upright)
    myvlm = Vlm(sail_set.panels, n_chordwise, n_spanwise, rho, w_profile, sail_set.trailing_edge_info, sail_set.leading_edge_info)

    inviscid_flow_results = InviscidFlowResults(sail_set, csys_transformations, myvlm)
    
    inviscid_flow_results.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)


    print("Preparing visualization.")   
    display_panels_xyz_and_winds(myvlm, inviscid_flow_results, myvlm.inlet_conditions, hull, show_plot=True)
    df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, csys_transformations, inviscid_flow_results, sail_set, output_dir_name, file_name)

    
    shutil.copy(os.path.join(case_dir, case_name), os.path.join(output_dir_name, case_name))

    print(f"-------------------------------------------------------------")
    print(f"Notice:\n"
          f"\tThe forces [N] and moments [Nm] are without profile drag.\n"
          f"\tThe the _COG_ CSYS is aligned in the direction of the yacht movement (course over ground).\n"
          f"\tThe the _COW_ CSYS is aligned along the centerline of the yacht (course over water).\n"
          f"\tNumber of panels (sail sail_set with mirror): {sail_set.panels.shape}")

    print(df_integrals)

    ##### 
    
    
    AR = 2 * main_sail_luff / main_sail_chords[0]
    S = 2*main_sail_luff * main_sail_chords[0]
    # prawdopodobnie jest zle układ
    # import numpy as np
    # >>> import sailing_vlm.runner.aircraft as ac
    # >>> a = ac.Aircraft(1.0, 5.0, 10.0, 32, 8, np.array([1.0, .0, .0]))
    # >>> a.get_Cxyz()
    # (0.023472780216173314, 0.0, 0.8546846987984326)

    # # Cx_vlm, Cy_vlm, Cz_vlm, total_F, V, S, q
    # (0.023472780216173314, 0.0, 0.8546846987984326, array([0.14377078, 0.        , 5.23494378]), array([1., 0., 0.]), 10.0, 6.125
    # metoda ponizej
    # 0.023472780216173342 -0.8546846987984335 6.996235308182204e-34 [ 1.43770779e-01 -5.23494378e+00  4.28519413e-33] [1. 0. 0.] 10.0 6.125
    CLx_vlm, Cy_vlm, Cz_vlm= get_vlm_Cxyz(myvlm.force, np.array(w_profile.get_true_wind_speed_at_h(1.0)), rho, S)
    print(f"C:[{CLx_vlm}, {Cy_vlm}, {Cz_vlm}]")#\nF_tot={tot_F}\nV={V} S={S} q={q}")
    print(f"AR: {AR}")
    print(f"S: {S}")

if __name__ == "__main__":
    
    start = time.time()
    cProfile.runctx('main()', {'main' : main}, {}, "output.dat")
    
    end = time.time()
    print("Elapsed = %s" % (end - start))

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()
        
    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()
        
    