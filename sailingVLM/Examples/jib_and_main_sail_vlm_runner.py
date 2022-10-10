import timeit
import shutil


from sailingVLM.YachtGeometry.SailFactory import SailFactory
from sailingVLM.YachtGeometry.SailGeometry import SailSet
from sailingVLM.Rotations.CSYS_transformations import CSYS_transformations
from sailingVLM.Solver.Interpolator import Interpolator
from sailingVLM.YachtGeometry.HullGeometry import HullGeometry
from sailingVLM.Inlet.InletConditions import InletConditions
from sailingVLM.Inlet.Winds import ExpWindProfile

from sailingVLM.ResultsContainers.save_results_utils import save_results_to_file
from sailingVLM.Solver.PanelsPlotter import display_panels_xyz_and_winds
from sailingVLM.Solver.vlm_solver import is_no_flux_BC_satisfied

from sailingVLM.Solver.vlm_solver import calc_circulation
from sailingVLM.ResultsContainers.InviscidFlowResults import prepare_inviscid_flow_results_vlm
from sailingVLM.Solver.vlm_solver import calculate_app_fs
from sailingVLM.ResultsContainers.InviscidFlowResults import InviscidFlowResults
from sailingVLM.Solver.forces import calc_force_VLM_xyz, calc_pressure

# from InputData.jib_and_main_sail_vlm_case_backflow import *
from InputData.jib_and_main_sail_vlm_case import *

# np.set_printoptions(precision=3, suppress=True)

start = timeit.default_timer()

interpolator = Interpolator(interpolation_type)

csys_transformations = CSYS_transformations(
    heel_deg, leeway_deg,
    v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

sail_factory = SailFactory(csys_transformations=csys_transformations, n_spanwise=n_spanwise, n_chordwise=n_chordwise,
                           rake_deg=rake_deg, sheer_above_waterline=sheer_above_waterline)

jib_geometry = sail_factory.make_jib(
    jib_luff=jib_luff,
    foretriangle_base=foretriangle_base,
    foretriangle_height=foretriangle_height,
    jib_chords=interpolator.interpolate_girths(jib_girths, jib_chords, n_spanwise + 1),
    sail_twist_deg=interpolator.interpolate_girths(jib_girths, jib_centerline_twist_deg, n_spanwise + 1),
    mast_LOA=mast_LOA,
    LLT_twist=LLT_twist)

main_sail_geometry = sail_factory.make_main_sail(
    main_sail_luff=main_sail_luff,
    boom_above_sheer=boom_above_sheer,
    main_sail_chords=interpolator.interpolate_girths(main_sail_girths, main_sail_chords, n_spanwise + 1),
    sail_twist_deg=interpolator.interpolate_girths(main_sail_girths, main_sail_centerline_twist_deg, n_spanwise + 1),
    LLT_twist=LLT_twist)

sail_set = SailSet([jib_geometry, main_sail_geometry])
# sail_set = SailSet([jib_geometry])

# wind = FlatWindProfile(alpha_true_wind_deg, tws_ref, SOG_yacht)
wind = ExpWindProfile(
    alpha_true_wind_deg, tws_ref, SOG_yacht,
    exp_coeff=wind_exp_coeff,
    reference_measurment_height=wind_reference_measurment_height,
    reference_water_level_for_wind_profile=reference_water_level_for_wind_profile)

inlet_condition = InletConditions(wind, rho=rho, panels1D=sail_set.panels1d)

hull = HullGeometry(sheer_above_waterline, foretriangle_base, csys_transformations, center_of_lateral_resistance_upright)

gamma_magnitude, v_ind_coeff = calc_circulation(inlet_condition.V_app_infs, sail_set.panels)
V_induced_at_ctrl_p, V_app_fs_at_ctrl_p = calculate_app_fs(inlet_condition, v_ind_coeff, gamma_magnitude)

assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, sail_set.panels)

inviscid_flow_results = prepare_inviscid_flow_results_vlm(gamma_magnitude, sail_set, inlet_condition, csys_transformations)
inviscid_flow_results.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)

df_components, df_integrals, df_inlet_IC = save_results_to_file(inviscid_flow_results, None, inlet_condition, sail_set, output_dir_name)
shutil.copy(os.path.join(case_dir, case_name), os.path.join(output_dir_name, case_name))

print(f"-------------------------------------------------------------")
print(f"Notice:\n"
      f"\tThe forces [N] and moments [Nm] are without profile drag.\n"
      f"\tThe the _COG_ CSYS is aligned in the direction of the yacht movement (course over ground).\n"
      f"\tThe the _COW_ CSYS is aligned along the centerline of the yacht (course over water).\n"
      f"\tNumber of panels (sail set with mirror): {sail_set.panels.shape}")

print(df_integrals)

# rows_to_display = ['M_total_heeling', 'M_total_sway', 'F_sails_drag'] # select rows to print
# print(df_integrals[df_integrals['Quantity'].isin(rows_to_display)])

print(f"\nCPU time: {float(timeit.default_timer() - start):.2f} [s]")

print("Preparing visualization.")
display_panels_xyz_and_winds(sail_set.panels1d, inlet_condition, inviscid_flow_results, hull)
print("Done.")
