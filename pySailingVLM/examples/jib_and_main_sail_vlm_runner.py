import timeit
import shutil

from pySailingVLM.yacht_geometry.sail_factory import SailFactory
from pySailingVLM.yacht_geometry.sail_geometry import SailSet
from pySailingVLM.rotations.csys_transformations import CSYS_transformations
from pySailingVLM.solver.interpolator import Interpolator
from pySailingVLM.yacht_geometry.hull_geometry import HullGeometry
from pySailingVLM.inlet.winds import ExpWindProfile

from pySailingVLM.results.save_utils import save_results_to_file
from pySailingVLM.solver.panels_plotter import display_panels_xyz_and_winds


from pySailingVLM.results.inviscid_flow import prepare_inviscid_flow_results_vlm

from pySailingVLM.examples.input_data.jib_and_main_sail_vlm_case import *
from pySailingVLM.solver.vlm import Vlm


                                      

def jib_runner():
        
    start = timeit.default_timer()

    interpolator = Interpolator(interpolation_type)

    csys_transformations = CSYS_transformations(
        heel_deg, leeway_deg,
        v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

    factory = SailFactory(csys_transformations=csys_transformations, n_spanwise=n_spanwise, n_chordwise=n_chordwise,
                            rake_deg=rake_deg, sheer_above_waterline=sheer_above_waterline)

    jib_geometry = factory.make_jib(
        jib_luff=jib_luff,
        foretriangle_base=foretriangle_base,
        foretriangle_height=foretriangle_height,
        jib_chords=interpolator.interpolate_girths(jib_girths, jib_chords, n_spanwise + 1),
        sail_twist_deg=interpolator.interpolate_girths(jib_girths, jib_centerline_twist_deg, n_spanwise + 1),
        mast_LOA=mast_LOA,
        LLT_twist=LLT_twist)

    main_sail_geometry = factory.make_main_sail(
        main_sail_luff=main_sail_luff,
        boom_above_sheer=boom_above_sheer,
        main_sail_chords=interpolator.interpolate_girths(main_sail_girths, main_sail_chords, n_spanwise + 1),
        sail_twist_deg=interpolator.interpolate_girths(main_sail_girths, main_sail_centerline_twist_deg, n_spanwise + 1),
        LLT_twist=LLT_twist)

    sail_set = SailSet([jib_geometry, main_sail_geometry])


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
    display_panels_xyz_and_winds(myvlm, inviscid_flow_results   , myvlm.inlet_conditions, hull, show_plot=True)


    df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, csys_transformations, inviscid_flow_results, sail_set, output_dir_name)

    
    shutil.copy(os.path.join(case_dir, case_name), os.path.join(output_dir_name, case_name))

    print(f"-------------------------------------------------------------")
    print(f"Notice:\n"
          f"\tThe forces [N] and moments [Nm] are without profile drag.\n"
          f"\tThe the _COG_ CSYS is aligned in the direction of the yacht movement (course over ground).\n"
          f"\tThe the _COW_ CSYS is aligned along the centerline of the yacht (course over water).\n"
          f"\tNumber of panels (sail sail_set with mirror): {sail_set.panels.shape}")

    print(df_integrals)

    print(f"\nCPU time: {float(timeit.default_timer() - start):.2f} [s]")
    


jib_runner()