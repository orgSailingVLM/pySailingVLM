import timeit
import shutil

from sailing_vlm.yacht_geometry.sail_factory import SailFactory
from sailing_vlm.yacht_geometry.sail_geometry import SailSet
from sailing_vlm.rotations.csys_transformations import CSYS_transformations
from sailing_vlm.solver.interpolator import Interpolator
from sailing_vlm.yacht_geometry.hull_geometry import HullGeometry
from sailing_vlm.inlet.winds import ExpWindProfile

from sailing_vlm.results.save_utils import save_results_to_file
from sailing_vlm.solver.panels_plotter import display_panels_xyz_and_winds


from sailing_vlm.results.inviscid_flow import prepare_inviscid_flow_results_vlm

from sailing_vlm.examples.input_data.jib_and_main_sail_vlm_case import *
from sailing_vlm.solver.vlm import Vlm

# czy to potrzebne???
from sailing_vlm.solver.forces import is_no_flux_BC_satisfied
from sailing_vlm.solver.velocity import calculate_app_fs

                                      

def main():
        
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

    set = SailSet([jib_geometry, main_sail_geometry])


    wind = ExpWindProfile(
        alpha_true_wind_deg, tws_ref, SOG_yacht,
        exp_coeff=wind_exp_coeff,
        reference_measurment_height=wind_reference_measurment_height,
        reference_water_level_for_wind_profile=reference_water_level_for_wind_profile)


    hull = HullGeometry(sheer_above_waterline, foretriangle_base, csys_transformations, center_of_lateral_resistance_upright)

    
    myvlm = Vlm(set.panels, n_chordwise, n_spanwise, rho, wind, set.sails, set.trailing_edge_info, set.leading_edge_info)


    inviscid_flow_results = prepare_inviscid_flow_results_vlm(set, csys_transformations, myvlm)
    inviscid_flow_results.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)


    print("Preparing visualization.")   
    display_panels_xyz_and_winds(myvlm, inviscid_flow_results   , myvlm.inlet_conditions, hull, show_plot=True)


    #todo
    # te rzeczy co sa zwracane sa na razie puste albo jakies bezsensowne
    df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, csys_transformations, inviscid_flow_results , myvlm.inlet_conditions, set, output_dir_name)

    #df_components, df_integrals, df_inlet_IC = save_results_to_file(inviscid_flow_results, None, inlet_condition, sail_set, output_dir_name)
    #new_df_components, new_df_integrals, new_df_inlet_IC = save_results_to_file(inviscid_flow_results_new_approach, None, myvlm.inlet_conditions, sail_set, output_dir_name)
    # shutil.copy(os.path.join(case_dir, case_name), os.path.join(output_dir_name, case_name))

    # print(f"-------------------------------------------------------------")
    # print(f"Notice:\n"
    #       f"\tThe forces [N] and moments [Nm] are without profile drag.\n"
    #       f"\tThe the _COG_ CSYS is aligned in the direction of the yacht movement (course over ground).\n"
    #       f"\tThe the _COW_ CSYS is aligned along the centerline of the yacht (course over water).\n"
    #       f"\tNumber of panels (sail set with mirror): {sail_set.panels.shape}")

    # print(df_integrals)

    # # rows_to_display = ['M_total_heeling', 'M_total_sway', 'F_sails_drag'] # select rows to print
    # # print(df_integrals[df_integrals['Quantity'].isin(rows_to_display)])

    # print(f"\nCPU time: {float(timeit.default_timer() - start):.2f} [s]")
    # #
    # # import matplotlib.pyplot as plt
    # # plt.plot([1,2,3],[5,6,7])
    # # plt.show()
    
if __name__ == "__main__":
    main()