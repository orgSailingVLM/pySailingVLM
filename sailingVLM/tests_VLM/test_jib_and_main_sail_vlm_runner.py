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
from sailingVLM.ResultsContainers.InviscidFlowResults import prepare_inviscid_flow_results_llt
from sailingVLM.Solver.vlm_solver import calculate_app_fs


import pandas as pd
from pandas.util.testing import assert_frame_equal
from unittest import TestCase


from sailingVLM.tests_VLM.InputFiles.case_data_for_vlm_runner import *
# np.set_printoptions(precision=3, suppress=True)


class TestVLM_Solver(TestCase):
    def setUp(self):
        interpolator = Interpolator(interpolation_type)

        self.csys_transformations = CSYS_transformations(
            heel_deg, leeway_deg,
            v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

        sail_factory = SailFactory(n_spanwise=n_spanwise, n_chordwise=n_chordwise, csys_transformations=self.csys_transformations, rake_deg=rake_deg,
                                   sheer_above_waterline=sheer_above_waterline)

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

        self.sail_set = SailSet([jib_geometry, main_sail_geometry])
        # sail_set = SailSet([jib_geometry])

        # wind = FlatWindProfile(alpha_true_wind_deg, tws_ref, SOG_yacht)
        wind = ExpWindProfile(
            alpha_true_wind_deg, tws_ref, SOG_yacht,
            exp_coeff=wind_exp_coeff,
            reference_measurment_height=wind_reference_measurment_height,
            reference_water_level_for_wind_profile=reference_water_level_for_wind_profile)

        self.inlet_condition = InletConditions(wind, rho=rho, panels1D=self.sail_set.panels1d)
        self.hull = HullGeometry(sheer_above_waterline, foretriangle_base, self.csys_transformations, center_of_lateral_resistance_upright)

    def test_calc_forces_and_moments(self):
        gamma_magnitude, v_ind_coeff, A = calc_circulation(self.inlet_condition.V_app_infs, self.sail_set.panels1d)
        V_induced, V_app_fs = calculate_app_fs(self.inlet_condition, v_ind_coeff, gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs, self.sail_set.panels1d)

        inviscid_flow_results = prepare_inviscid_flow_results_llt(V_app_fs, V_induced, gamma_magnitude,
                                                                  self.sail_set, self.inlet_condition,
                                                                  self.csys_transformations)

        inviscid_flow_results.estimate_heeling_moment_from_keel(self.hull.center_of_lateral_resistance)
        display_panels_xyz_and_winds(self.sail_set.panels1d, self.inlet_condition, inviscid_flow_results, self.hull, show_plot=False)

        df_components, df_integrals, df_inlet_IC = save_results_to_file(
            inviscid_flow_results, None, self.inlet_condition, self.sail_set, output_dir_name)
        shutil.copy(os.path.join(case_dir, case_name), os.path.join(output_dir_name, case_name))

        # print(f"-------------------------------------------------------------")
        # print(f"Notice:\n"
        #       f"\tThe forces [N] and moments [Nm] are without profile drag.\n"
        #       f"\tThe the _COG_ CSYS is aligned in the direction of the yacht movement (course over ground).\n"
        #       f"\tThe the _COW_ CSYS is aligned along the centerline of the yacht (course over water).\n")
        #

        df_components.to_csv('expected_df_components.csv')
        df_integrals.to_csv('expected_df_integrals.csv', index=False)
        df_inlet_IC.to_csv('expected_df_inlet_IC.csv')

        expected_df_components = pd.read_csv(os.path.join(case_dir, 'expected_df_components.csv'))
        expected_df_components.set_index('Unnamed: 0', inplace=True)  # the mirror part is not stored, thus half of the indices are cut off
        expected_df_components.index.name = None
        assert_frame_equal(df_components, expected_df_components)

        expected_df_integrals = pd.read_csv(os.path.join(case_dir, 'expected_df_integrals.csv'))
        assert_frame_equal(df_integrals, expected_df_integrals)

        expected_df_inlet_ic = pd.read_csv(os.path.join(case_dir, 'expected_df_inlet_IC.csv'))
        expected_df_inlet_ic.set_index('Unnamed: 0', inplace=True)  # the mirror part is not stored, thus half of the indices are cut off
        expected_df_inlet_ic.index.name = None
        assert_frame_equal(df_inlet_IC, expected_df_inlet_ic)
