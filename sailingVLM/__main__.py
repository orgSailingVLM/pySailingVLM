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
from sailingVLM.Solver.PanelsPlotter import display_panels_xyz_and_winds, display_panels_xyz_and_winds_new_approach
from sailingVLM.Solver.vlm_solver import is_no_flux_BC_satisfied

from sailingVLM.Solver.vlm_solver import calc_circulation
from sailingVLM.ResultsContainers.InviscidFlowResults import prepare_inviscid_flow_results_vlm, prepare_inviscid_flow_results_vlm_new_approach
from sailingVLM.Solver.vlm_solver import calculate_app_fs
from sailingVLM.ResultsContainers.InviscidFlowResults import InviscidFlowResults

# from InputData.jib_and_main_sail_vlm_case_backflow import *
from sailingVLM.Examples.InputData.jib_and_main_sail_vlm_case import *

from sailingVLM.NewApproach.vlm import NewVlm
from sailingVLM.Solver.TrailingEdgePanel import TrailingEdgePanel
###
#from sailingVLM.NewApproach.vlm_logic import get_panels_area, \
#                                            calculate_normals_collocations_cps_rings_spans, \
#                                            get_influence_coefficients_spanwise, \
#                                            solve_eq, \
#                                            get_influence_coefficients_spanwise_jib_version,

import sailingVLM.NewApproach.vlm_logic as vlm_logic 
from unittest import TestCase                                       
###
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


# wind = FlatWindProfile(alpha_true_wind_deg, tws_ref, SOG_yacht)
wind = ExpWindProfile(
    alpha_true_wind_deg, tws_ref, SOG_yacht,
    exp_coeff=wind_exp_coeff,
    reference_measurment_height=wind_reference_measurment_height,
    reference_water_level_for_wind_profile=reference_water_level_for_wind_profile)

inlet_condition = InletConditions(wind, rho=rho, panels1D=sail_set.panels1d)



hull = HullGeometry(sheer_above_waterline, foretriangle_base, csys_transformations, center_of_lateral_resistance_upright)


cp_good = []
ctr_good = []
normals_good = []
area_good = []
spans_good = []
rings_good = []
gammas_good = []
panels_good = []
horseshoe_info_panels = []
for item in sail_set.panels:
    for panel in item:
        panels_good.append(panel.get_points())
        cp_good.append(panel.cp_position)
        ctr_good.append(panel.get_ctr_point_position())
        normals_good.append(panel.get_normal_to_panel())
        area_good.append(panel.get_panel_area())
        spans_good.append(panel.get_span_vector())
        rings_good.append(panel.get_vortex_ring_position())
        # checking gamma orientation
        gammas_good.append(panel.gamma_orientation)
        
        
        if isinstance(panel, TrailingEdgePanel):
            horseshoe_info_panels.append(True)
        else:
            horseshoe_info_panels.append(False)
        
        
        

cp_good = np.array(cp_good)
ctr_good = np.array(ctr_good)
normals_good = np.array(normals_good)
area_good = np.array(area_good)
spans_good = np.array(spans_good)
rings_good = np.array(rings_good)
# wyszly wszystkie minus jedynki
gammas_good = np.array(gammas_good)
panels_good = np.array(panels_good)

# not used?
horseshoe_info_panels = np.array(horseshoe_info_panels)


gamma_orientation = -1
myvlm = NewVlm(sail_set.my_panels, n_chordwise, n_spanwise, rho, wind, sail_set.sails, sail_set.trailing_edge_info, sail_set.leading_edge_info, gamma_orientation)

#my_inlet_condition = InletConditions(wind, rho=rho, panels1D=myvlm.panels)

# sortowanie jest bo inaczej nie porownam tego bo mam inny uklad paneli
np.testing.assert_almost_equal(np.sort(panels_good, axis=0), np.sort(myvlm.panels, axis=0))
np.testing.assert_almost_equal(np.sort(cp_good, axis=0), np.sort(myvlm.center_of_pressure, axis=0))
np.testing.assert_almost_equal(np.sort(ctr_good, axis=0), np.sort(myvlm.collocation_points, axis=0))
np.testing.assert_almost_equal(np.sort(normals_good, axis=0), np.sort(myvlm.normals, axis=0))
#area_good = area_good.reshape(area_good.shape[0],1)
np.testing.assert_almost_equal(np.sort(area_good, axis=0), np.sort(myvlm.areas, axis=0))
np.testing.assert_almost_equal(np.sort(spans_good, axis=0), np.sort(myvlm.span_vectors, axis=0))
np.testing.assert_almost_equal(np.sort(rings_good, axis=0), np.sort(myvlm.rings, axis=0))



gamma_magnitude, v_ind_coeff, A, RHS_good = calc_circulation(inlet_condition.V_app_infs, sail_set.panels)

np.testing.assert_almost_equal(np.sort(RHS_good, axis=0), np.sort(myvlm.RHS, axis=0))

V_induced_at_ctrl_p, V_app_fs_at_ctrl_p = calculate_app_fs(inlet_condition.V_app_infs, v_ind_coeff, gamma_magnitude)


V_induced_at_ctrl_p_my, V_app_fs_at_ctrl_p_my = calculate_app_fs(myvlm.inlet_conditions.V_app_infs, myvlm.wind_coefs, myvlm.gamma_magnitude)
np.testing.assert_almost_equal(np.sort(V_induced_at_ctrl_p, axis=0), np.sort(V_induced_at_ctrl_p_my, axis=0))
np.testing.assert_almost_equal(np.sort(V_app_fs_at_ctrl_p, axis=0), np.sort(V_app_fs_at_ctrl_p_my, axis=0))



assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, sail_set.panels)
assert vlm_logic.is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p_my, myvlm.panels, myvlm.areas, myvlm.normals)

# to be fixed
inviscid_flow_results = prepare_inviscid_flow_results_vlm(gamma_magnitude, sail_set, inlet_condition, csys_transformations, myvlm)
inviscid_flow_results_new_approach = prepare_inviscid_flow_results_vlm_new_approach(sail_set, csys_transformations, myvlm)

np.testing.assert_almost_equal(np.sort(inviscid_flow_results.gamma_magnitude, axis=0), np.sort(inviscid_flow_results_new_approach.gamma_magnitude, axis=0))
# sprawdzic to
#np.testing.assert_almost_equal(np.sort(inviscid_flow_results.csys_transformations, axis=0), np.sort(inviscid_flow_results_new_approach.csys_transformations, axis=0))

np.testing.assert_almost_equal(np.sort(inviscid_flow_results.pressure, axis=0), np.sort(myvlm.pressure, axis=0))

np.testing.assert_almost_equal(np.sort(inviscid_flow_results.pressure, axis=0), np.sort(inviscid_flow_results_new_approach.pressure, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.V_induced_at_cp, axis=0), np.sort(inviscid_flow_results_new_approach.V_induced_at_cp, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.V_induced_length, axis=0), np.sort(inviscid_flow_results_new_approach.V_induced_length, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.V_app_fs_at_cp, axis=0), np.sort(inviscid_flow_results_new_approach.V_app_fs_at_cp, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.V_app_fs_length, axis=0), np.sort(inviscid_flow_results_new_approach.V_app_fs_length, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.AWA_app_fs, axis=0), np.sort(inviscid_flow_results_new_approach.AWA_app_fs, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.F_xyz, axis=0), np.sort(inviscid_flow_results_new_approach.F_xyz, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.F_xyz_total, axis=0), np.sort(inviscid_flow_results_new_approach.F_xyz_total, axis=0))
# ogarnac
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.F_xyz_above_water, axis=0), np.sort(inviscid_flow_results_new_approach.F_xyz_above_water, axis=0))

np.testing.assert_almost_equal(np.sort(inviscid_flow_results.r, axis=0), np.sort(inviscid_flow_results_new_approach.r, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.r_above_water, axis=0), np.sort(inviscid_flow_results_new_approach.r_above_water, axis=0))

np.testing.assert_almost_equal(np.sort(inviscid_flow_results.r_above_water, axis=0), np.sort(inviscid_flow_results_new_approach.r_above_water, axis=0))

np.testing.assert_almost_equal(np.sort(inviscid_flow_results.M_xyz, axis=0), np.sort(inviscid_flow_results_new_approach.M_xyz, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.M_total_above_water_in_xyz_csys, axis=0), np.sort(inviscid_flow_results_new_approach.M_total_above_water_in_xyz_csys, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.above_water_centre_of_effort_estimate_xyz, axis=0), np.sort(inviscid_flow_results_new_approach.above_water_centre_of_effort_estimate_xyz, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.F_centerline, axis=0), np.sort(inviscid_flow_results_new_approach.F_centerline, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.F_centerline_total, axis=0), np.sort(inviscid_flow_results_new_approach.F_centerline_total, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.M_centerline_csys, axis=0), np.sort(inviscid_flow_results_new_approach.M_centerline_csys, axis=0))
np.testing.assert_almost_equal(np.sort(inviscid_flow_results.M_total_above_water_in_centerline_csys, axis=0), np.sort(inviscid_flow_results_new_approach.M_total_above_water_in_centerline_csys, axis=0))


np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['F_jib_total_COG.x'],inviscid_flow_results_new_approach.dyn_dict['F_jib_total_COG.x'])
np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['F_jib_total_COG.y'],inviscid_flow_results_new_approach.dyn_dict['F_jib_total_COG.y'])
np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['F_jib_total_COG.z'],inviscid_flow_results_new_approach.dyn_dict['F_jib_total_COG.z'])

np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['F_main_sail_total_COG.x'],inviscid_flow_results_new_approach.dyn_dict['F_main_sail_total_COG.x'])
np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['F_main_sail_total_COG.y'],inviscid_flow_results_new_approach.dyn_dict['F_main_sail_total_COG.y'])
np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['F_main_sail_total_COG.z'],inviscid_flow_results_new_approach.dyn_dict['F_main_sail_total_COG.z'])



np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['M_jib_total_COG.x'],inviscid_flow_results_new_approach.dyn_dict['M_jib_total_COG.x'])
np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['M_jib_total_COG.y'],inviscid_flow_results_new_approach.dyn_dict['M_jib_total_COG.y'])
np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['M_jib_total_COG.z'],inviscid_flow_results_new_approach.dyn_dict['M_jib_total_COG.z'])

np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['M_main_sail_total_COG.x'],inviscid_flow_results_new_approach.dyn_dict['M_main_sail_total_COG.x'])
np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['M_main_sail_total_COG.y'],inviscid_flow_results_new_approach.dyn_dict['M_main_sail_total_COG.y'])
np.testing.assert_almost_equal(inviscid_flow_results.dyn_dict['M_main_sail_total_COG.z'],inviscid_flow_results_new_approach.dyn_dict['M_main_sail_total_COG.z'])



###


#dyn_dict
inviscid_flow_results.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)
inviscid_flow_results_new_approach.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)


print("Preparing visualization.")
# good, uncomment and fix boolean to not do the plot
#display_panels_xyz_and_winds(myvlm, inviscid_flow_results_new_approach, sail_set.panels1d, inlet_condition, myvlm.inlet_conditions, inviscid_flow_results, hull, show_plot=False)


# tu spawdzic
# assert zrobic, na bank nie przejdzie bo save_results_to_file needs changes
df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, csys_transformations, inviscid_flow_results, inviscid_flow_results_new_approach, inlet_condition, myvlm.inlet_conditions, sail_set, output_dir_name)

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