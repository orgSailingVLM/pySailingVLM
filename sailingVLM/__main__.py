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
from sailingVLM.Examples.InputData.jib_and_main_sail_vlm_case import *

from sailingVLM.NewApproach.vlm import NewVlm
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

# zminić pod siebie
# panele ktore maja isc do mojej cyrkulacji powinny byc takie (1d array)
###
gamma_orientation = -1
myvlm = NewVlm(sail_set.sail_panels, n_chordwise, n_spanwise, inlet_condition.rho, inlet_condition.V_app_infs, sail_set.sails, sail_set.horseshoe_info, gamma_orientation)

cp_good = []
ctr_good = []
normals_good = []
area_good = []
spans_good = []
rings_good = []
gammas_good = []
panels_good = []
for item in sail_set.panels:
    for panel in item:
        panels_good.append(panel.get_points())
        cp_good.append(panel.get_cp_position())
        ctr_good.append(panel.get_ctr_point_position())
        normals_good.append(panel.get_normal_to_panel())
        area_good.append(panel.get_panel_area())
        spans_good.append(panel.get_span_vector())
        rings_good.append(panel.get_vortex_ring_position())
        # checking gamma orientation
        gammas_good.append(panel.gamma_orientation)
        
        

cp_good = np.array(cp_good)
ctr_good = np.array(ctr_good)
normals_good = np.array(normals_good)
area_good = np.array(area_good)
spans_good = np.array(spans_good)
rings_good = np.array(rings_good)
# wyszly wszystkie minus jedynki
gammas_good = np.array(gammas_good)
panels_good = np.array(panels_good)


# np.testing.assert_almost_equal(cp_good, myvlm.center_of_pressure)
# np.testing.assert_almost_equal(ctr_good, myvlm.collocation_points)
# np.testing.assert_almost_equal(normals_good, myvlm.normals)
# area_good = area_good.reshape(area_good.shape[0],1)
# np.testing.assert_almost_equal(area_good, myvlm.areas)
# np.testing.assert_almost_equal(spans_good, myvlm.span_vectors)
# np.testing.assert_almost_equal(rings_good, myvlm.rings)
case = TestCase()
# tests if elements  are the same egardles of order
case.assertCountEqual(myvlm.panels.tolist(), panels_good.tolist())
case.assertCountEqual(cp_good.tolist(), myvlm.center_of_pressure.tolist())
case.assertCountEqual(ctr_good.tolist(), myvlm.collocation_points.tolist())
case.assertCountEqual(normals_good.tolist(), myvlm.normals.tolist())
area_good = area_good.reshape(area_good.shape[0],1)
case.assertCountEqual(area_good.tolist(), myvlm.areas.tolist())
case.assertCountEqual(spans_good.tolist(), myvlm.span_vectors.tolist())
case.assertCountEqual(rings_good.tolist(), myvlm.rings.tolist())



####
gamma_magnitude, v_ind_coeff, A, RHS_good = calc_circulation(inlet_condition.V_app_infs, sail_set.panels)
# tu do poprawki
# zle jest liczny horseshoe 
# patrz linijka 174 i 240 SailGeometry.py
case.assertCountEqual(RHS_good.tolist(), myvlm.RHS.tolist())

# WIELKI TEST
case.assertCountEqual(A.tolist(), myvlm.coefs.tolist())
case.assertCountEqual(myvlm.wind_coefs.tolist(), v_ind_coeff.tolist())


V_induced_at_ctrl_p, V_app_fs_at_ctrl_p = calculate_app_fs(inlet_condition.V_app_infs, v_ind_coeff, gamma_magnitude)


V_induced_at_ctrl_p_my, V_app_fs_at_ctrl_p_my = calculate_app_fs(inlet_condition.V_app_infs, myvlm.wind_coefs, myvlm.gamma_magnitude)
case.assertCountEqual(V_induced_at_ctrl_p.tolist(), V_induced_at_ctrl_p_my.tolist())
case.assertCountEqual(V_app_fs_at_ctrl_p.tolist(), V_app_fs_at_ctrl_p_my.tolist())


# to zawarte jest w NewVLM
assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, sail_set.panels)

# my
# popr
assert vlm_logic.is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p_my, sail_set.my_panels, myvlm.areas, myvlm.normals)

# to be fixed
inviscid_flow_results = prepare_inviscid_flow_results_vlm(gamma_magnitude, sail_set, inlet_condition, csys_transformations, myvlm)
inviscid_flow_results.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)

print("Preparing visualization.")
display_panels_xyz_and_winds(sail_set.panels1d, inlet_condition, inviscid_flow_results, hull)

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
#
# import matplotlib.pyplot as plt
# plt.plot([1,2,3],[5,6,7])
# plt.show()