import timeit
import shutil

from sailing_vlm.yacht_geometry.sail_factory import SailFactory
from sailing_vlm.yacht_geometry.sail_geometry import SailSet
from sailing_vlm.rotations.csys_transformations import CSYS_transformations
from sailing_vlm.solver.interpolator import Interpolator
from sailing_vlm.yacht_geometry.hull_geometry import HullGeometry
from sailing_vlm.inlet.winds import ExpWindProfile

from sailing_vlm.results.save_utils import save_results_to_file
from sailing_vlm.solver.panels_plotter import display_panels_xyz_and_winds, display_panels_or_rings


from sailing_vlm.results.inviscid_flow import prepare_inviscid_flow_results_vlm

from sailing_vlm.examples.input_data.jib_and_main_sail_vlm_case import *
from sailing_vlm.solver.vlm import Vlm


import pstats
import numpy as np
import cProfile
import time
import sys, os
from pstats import SortKey
from contextlib import redirect_stdout
### GEOMETRY DEFINITION ###



def main():
    
    
        
    try:
        assert len(jib_girths) == len(jib_chords), "ERROR!: jib_girths array must have same size as jib_chords!"
        assert len(jib_girths) == len(jib_centerline_twist_deg), "ERROR!: jib_girths array must have same size as jib_centerline_twist_deg!"
        assert len(jib_girths) == len(jib_sail_camber), "ERROR!: jib_girths array must have same size as jib_sail_camber!"
        assert len(jib_girths) == len(jib_sail_camber_distance_from_LE), "ERROR!: jib_girths array must have same size as jib_sail_camber_distance_from_LE!"
        
        assert len(main_sail_girths) == len(main_sail_chords), "ERROR!: main_sail_girths array must have same size as main_sail_chords!"
        assert len(main_sail_girths) == len(main_sail_centerline_twist_deg), "ERROR!: main_sail_girths array must have same size as main_sail_centerline_twist_deg!"
        assert len(main_sail_girths) == len(main_sail_camber), "ERROR!: main_sail_girths array must have same size as main_sail_camber!"
        assert len(main_sail_girths) == len(main_sail_camber_distance_from_LE), "ERROR!: main_sail_girths array must have same size as main_sail_camber_distance_from_LE!"
        
        allowed_sails_def = ['jib', 'main', 'jib_and_main']
        
        if sails_def not in allowed_sails_def:
            raise ValueError(f'ERROR!: {sails_def} not allowed as sails_def variable!')
            
    except (AssertionError, ValueError) as err:
        print(err)
        sys.exit()
    
    interpolator = Interpolator(interpolation_type)

    csys_transformations = CSYS_transformations(
        heel_deg, leeway_deg,
        v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

    factory = SailFactory(csys_transformations=csys_transformations, n_spanwise=n_spanwise, n_chordwise=n_chordwise,
                            rake_deg=rake_deg, sheer_above_waterline=sheer_above_waterline)
    
    geoms = []
    if sails_def == 'jib' or sails_def == 'jib_and_main':
        jib_geometry = factory.make_jib(
            jib_luff=jib_luff,
            foretriangle_base=foretriangle_base,
            foretriangle_height=foretriangle_height,
            jib_chords=interpolator.interpolate_girths(jib_girths, jib_chords, n_spanwise + 1),
            sail_twist_deg=interpolator.interpolate_girths(jib_girths, jib_centerline_twist_deg,n_spanwise + 1),
            mast_LOA=mast_LOA,
            LLT_twist=LLT_twist, 
            interpolated_camber=interpolator.interpolate_girths(jib_girths, jib_sail_camber, n_spanwise + 1),
            interpolated_distance_from_LE=interpolator.interpolate_girths(jib_girths, jib_sail_camber_distance_from_LE, n_spanwise + 1)
            )
        geoms.append(jib_geometry)
        
    if sails_def == 'main' or sails_def =='jib_and_main':
        main_sail_geometry = factory.make_main_sail(
            main_sail_luff=main_sail_luff,
            boom_above_sheer=boom_above_sheer,
            main_sail_chords=interpolator.interpolate_girths(main_sail_girths, main_sail_chords, n_spanwise + 1),
            sail_twist_deg=interpolator.interpolate_girths(main_sail_girths, main_sail_centerline_twist_deg, n_spanwise + 1),
            LLT_twist=LLT_twist,
            interpolated_camber=interpolator.interpolate_girths(main_sail_girths, main_sail_camber, n_spanwise + 1),
            interpolated_distance_from_LE=interpolator.interpolate_girths(main_sail_girths, main_sail_camber_distance_from_LE, n_spanwise + 1)
            )
        geoms.append(main_sail_geometry)

    sail_set = SailSet(geoms)

    wind = ExpWindProfile(
        alpha_true_wind_deg, tws_ref, SOG_yacht,
        exp_coeff=wind_exp_coeff,
        reference_measurment_height=wind_reference_measurment_height,
        reference_water_level_for_wind_profile=reference_water_level_for_wind_profile)


    hull = HullGeometry(sheer_above_waterline, foretriangle_base, csys_transformations, center_of_lateral_resistance_upright)

    
    myvlm = Vlm(sail_set.panels, n_chordwise, n_spanwise, rho, wind, sail_set.trailing_edge_info, sail_set.leading_edge_info)


    inviscid_flow_results = prepare_inviscid_flow_results_vlm(sail_set, csys_transformations, myvlm)
    inviscid_flow_results.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)


    print("Preparing visualization.")   
    display_panels_xyz_and_winds(myvlm, inviscid_flow_results, myvlm.inlet_conditions, hull, show_plot=True)
    #display_panels_or_rings(myvlm.rings, myvlm.pressure, myvlm.leading_mid_points, myvlm.trailing_mid_points)
    df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, csys_transformations, inviscid_flow_results, sail_set, output_dir_name)

    
    shutil.copy(os.path.join(case_dir, case_name), os.path.join(output_dir_name, case_name))

    print(f"-------------------------------------------------------------")
    print(f"Notice:\n"
          f"\tThe forces [N] and moments [Nm] are without profile drag.\n"
          f"\tThe the _COG_ CSYS is aligned in the direction of the yacht movement (course over ground).\n"
          f"\tThe the _COW_ CSYS is aligned along the centerline of the yacht (course over water).\n"
          f"\tNumber of panels (sail sail_set with mirror): {sail_set.panels.shape}")

    print(df_integrals)

    
    
    
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
        
    