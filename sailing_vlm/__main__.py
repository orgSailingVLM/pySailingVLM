import timeit
import shutil

from varname import nameof

from sailing_vlm.yacht_geometry.sail_factory import SailFactory
from sailing_vlm.yacht_geometry.sail_geometry import SailSet
from sailing_vlm.rotations.csys_transformations import CSYS_transformations
from sailing_vlm.solver.interpolator import Interpolator
from sailing_vlm.yacht_geometry.hull_geometry import HullGeometry
from sailing_vlm.inlet.winds import ExpWindProfile, FlatWindProfile, LogWindProfile

from sailing_vlm.results.save_utils import save_results_to_file
from sailing_vlm.solver.panels_plotter import display_panels_xyz_and_winds, display_panels_or_rings


from sailing_vlm.results.inviscid_flow import InviscidFlowResults

from sailing_vlm.examples.input_data.jib_and_main_sail_vlm_case import *
from sailing_vlm.solver.vlm import Vlm


import pstats
import numpy as np
import cProfile
import time
import sys, os



def check_var(var: str, allowed_vars : list, var_name : str):
    """
    check_var check if variable is allowed

    :param str var: variable to check
    :param list allowed_vars: list with allowed variables
    :raises ValueError: raises error message for user
    """
    if var not in allowed_vars:
        raise ValueError(f'ERROR!: {var} not allowed as {var_name} variable!')

def assert_array_input(arr1 : np.array, arr2 : np.array, name_arr1: str, name_arr2 : str):
    """
    assert_array_input check if arrays hase the same length and print error message

    :param np.array arr1: arr1
    :param np.array arr2: arr2
    :param str name_arr1: name of arr1
    :param str name_arr2: name of arr2
    """
    assert len(arr1) == len(arr2), f'ERROR!: {name_arr1} array must have same size as {name_arr2}!'

def check_input_variables():
    try:
        assert_array_input(jib_girths, jib_chords, 'jib_girths', 'job_chords')
        assert_array_input(jib_girths, jib_centerline_twist_deg, 'jib_girths', 'jib_centerline_twist_deg')
        assert_array_input(jib_girths, jib_sail_camber, 'jib_girths', 'jib_sail_camber')
        assert_array_input(jib_girths, jib_sail_camber_distance_from_luff, 'jib_girths', 'jib_sail_camber_distance_from_luff')
        
        assert_array_input(main_sail_girths, main_sail_chords, 'main_sail_girths', 'main_sail_chords')
        assert_array_input(main_sail_girths, main_sail_centerline_twist_deg, 'main_sail_girths', 'main_sail_centerline_twist_deg')
        assert_array_input(main_sail_girths, main_sail_camber, 'main_sail_girths', 'main_sail_camber')
        assert_array_input(main_sail_girths, main_sail_camber_distance_from_luff, 'main_sail_girths', 'main_sail_camber_distance_from_luff')
        
        check_var(sails_def, ['jib', 'main', 'jib_and_main'], 'sails_def')
        check_var(interpolation_type, ['spline', 'linear'], 'interpolation_type')
        check_var(LLT_twist, ['real_twist', 'sheeting_angle_const', 'average_const'], 'LLT_twist')
        check_var(wind_profile, ['exponential', 'flat', 'logarithmic'], 'wind_profile')
        
    except (AssertionError, ValueError) as err:
        print(err)
        sys.exit()

def set_wind():
    if wind_profile == 'exponential':
        wind = ExpWindProfile(
            alpha_true_wind_deg, tws_ref, SOG_yacht,
            exp_coeff=wind_exp_coeff,
            reference_measurment_height=wind_reference_measurment_height,
            reference_water_level_for_wind_profile=reference_water_level_for_wind_profile)
    elif wind_profile == 'flat':
        wind = FlatWindProfile(alpha_true_wind_deg, tws_ref, SOG_yacht)
    else:
        wind = LogWindProfile(
            alpha_true_wind_deg, tws_ref, SOG_yacht,
            roughness=roughness,
            reference_measurment_height=wind_reference_measurment_height)
    return wind

def generate_sail_set(csys_transformations : CSYS_transformations) -> SailSet:
    interpolator = Interpolator(interpolation_type)
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
            interpolated_distance_from_luff=interpolator.interpolate_girths(jib_girths, jib_sail_camber_distance_from_luff, n_spanwise + 1)
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
            interpolated_distance_from_luff=interpolator.interpolate_girths(main_sail_girths, main_sail_camber_distance_from_luff, n_spanwise + 1)
            )
        geoms.append(main_sail_geometry)

    return SailSet(geoms)
    
def main():
    
    check_input_variables()

    csys_transformations = CSYS_transformations(
        heel_deg, leeway_deg,
        v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

    sail_set = generate_sail_set(csys_transformations)
    wind = set_wind()
    hull = HullGeometry(sheer_above_waterline, foretriangle_base, csys_transformations, center_of_lateral_resistance_upright)
    myvlm = Vlm(sail_set.panels, n_chordwise, n_spanwise, rho, wind, sail_set.trailing_edge_info, sail_set.leading_edge_info)

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
        
    