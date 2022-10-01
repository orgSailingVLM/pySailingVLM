import shutil

from YachtGeometry.SailFactory import SailFactory
from YachtGeometry.SailGeometry import SailSet
from Rotations.CSYS_transformations import CSYS_transformations
from Solver.Interpolator import Interpolator
from YachtGeometry.HullGeometry import HullGeometry
from Inlet.InletConditions import InletConditions
from Inlet.Winds import ExpWindProfile

from Solver.forces import calc_V_at_cp
from ResultsContainers.save_results_utils import save_results_to_file
from Solver.PanelsPlotter import display_panels_xyz_and_winds
from Solver.vlm_solver import is_no_flux_BC_satisfied

from Solver.vlm_solver import calc_circulation
from ResultsContainers.InviscidFlowResults import prepare_inviscid_flow_results_llt, prepare_inviscid_flow_results_vlm
from Solver.vlm_solver import calculate_app_fs

import pandas as pd
from pandas.util.testing import assert_frame_equal
from unittest import TestCase


from tests_VLM.InputFiles.case_data_for_vlm_runner import *
# np.set_printoptions(precision=3, suppress=True)


class TestVLM_Solver(TestCase):
    def setUp(self):
        self.interpolator = Interpolator(interpolation_type)

        self.csys_transformations = CSYS_transformations(
            heel_deg, leeway_deg,
            v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

        # wind = FlatWindProfile(alpha_true_wind_deg, tws_ref, SOG_yacht)
        self.wind = ExpWindProfile(
            alpha_true_wind_deg, tws_ref, SOG_yacht,
            exp_coeff=wind_exp_coeff,
            reference_measurment_height=wind_reference_measurment_height,
            reference_water_level_for_wind_profile=reference_water_level_for_wind_profile)


        self.hull = HullGeometry(sheer_above_waterline, foretriangle_base, self.csys_transformations, center_of_lateral_resistance_upright)

    def _prepare_sail_set(self, n_spanwise, n_chordwise):

        sail_factory = SailFactory(n_spanwise=n_spanwise, n_chordwise=n_chordwise,
                                   csys_transformations=self.csys_transformations, rake_deg=rake_deg,
                                   sheer_above_waterline=sheer_above_waterline)

        jib_geometry = sail_factory.make_jib(
            jib_luff=jib_luff,
            foretriangle_base=foretriangle_base,
            foretriangle_height=foretriangle_height,
            jib_chords=self.interpolator.interpolate_girths(jib_girths, jib_chords, n_spanwise + 1),
            sail_twist_deg=self.interpolator.interpolate_girths(jib_girths, jib_centerline_twist_deg, n_spanwise + 1),
            mast_LOA=mast_LOA,
            LLT_twist=LLT_twist)

        main_sail_geometry = sail_factory.make_main_sail(
            main_sail_luff=main_sail_luff,
            boom_above_sheer=boom_above_sheer,
            main_sail_chords=self.interpolator.interpolate_girths(main_sail_girths, main_sail_chords, n_spanwise + 1),
            sail_twist_deg=self.interpolator.interpolate_girths(main_sail_girths, main_sail_centerline_twist_deg, n_spanwise + 1),
            LLT_twist=LLT_twist)

        sail_set = SailSet([jib_geometry, main_sail_geometry])
        inlet_condition = InletConditions(self.wind, rho=rho, panels1D=sail_set.panels1d)
        return sail_set, inlet_condition




    def _check_df_results(self, suffix, inviscid_flow_results, inlet_condition, sail_set):
        inviscid_flow_results.estimate_heeling_moment_from_keel(self.hull.center_of_lateral_resistance)
        # display_panels_xyz_and_winds(self.sail_set.panels1d, self.inlet_condition, inviscid_flow_results, self.hull, show_plot=False)

        df_components, df_integrals, df_inlet_IC = save_results_to_file(
            inviscid_flow_results, None, inlet_condition, sail_set, output_dir_name)
        shutil.copy(os.path.join(case_dir, case_name), os.path.join(output_dir_name, case_name))


        df_components.to_csv(f'expected_df_components_{suffix}.csv')
        df_integrals.to_csv(f'expected_df_integrals_{suffix}.csv', index=False)
        df_inlet_IC.to_csv(f'expected_df_inlet_IC_{suffix}.csv')

        expected_df_integrals = pd.read_csv(os.path.join(case_dir, f'expected_df_integrals_{suffix}.csv'))
        assert_frame_equal(df_integrals, expected_df_integrals)

        expected_df_inlet_ic = pd.read_csv(os.path.join(case_dir, f'expected_df_inlet_IC_{suffix}.csv'))
        expected_df_inlet_ic.set_index('Unnamed: 0', inplace=True)  # the mirror part is not stored, thus half of the indices are cut off
        expected_df_inlet_ic.index.name = None
        assert_frame_equal(df_inlet_IC, expected_df_inlet_ic)

        expected_df_components = pd.read_csv(os.path.join(case_dir, f'expected_df_components_{suffix}.csv'))
        expected_df_components.set_index('Unnamed: 0', inplace=True)  # the mirror part is not stored, thus half of the indices are cut off
        expected_df_components.index.name = None
        assert_frame_equal(df_components, expected_df_components)

    def test_calc_forces_and_moments_vlm(self):
        suffix = 'vlm'
        n_chordwise = 3

        sail_set, inlet_condition = self._prepare_sail_set(n_spanwise=10, n_chordwise=n_chordwise)
        gamma_magnitude, v_ind_coeff = calc_circulation(inlet_condition.V_app_infs, sail_set.panels1d)

        df_gamma = pd.DataFrame({'gamma_magnitute': gamma_magnitude})
        df_gamma.to_csv(f'expected_gamma_magnitute_{suffix}.csv', index=False)
        expected_df_gamma = pd.read_csv(os.path.join(case_dir, f'expected_gamma_magnitute_{suffix}.csv'))
        assert_frame_equal(df_gamma, expected_df_gamma)

        V_induced_at_ctrl_p, V_app_fs_at_ctrl_p = calculate_app_fs(inlet_condition, v_ind_coeff, gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, sail_set.panels1d)
        inviscid_flow_results_vlm = prepare_inviscid_flow_results_vlm(gamma_magnitude,
                                                                   sail_set, inlet_condition,
                                                                   self.csys_transformations)

        self._check_df_results(suffix, inviscid_flow_results_vlm, inlet_condition, sail_set)

    def test_calc_forces_and_moments_llt(self):
        suffix = 'llt'
        n_chordwise = 1

        sail_set, inlet_condition = self._prepare_sail_set(n_spanwise=10, n_chordwise=n_chordwise)
        gamma_magnitude, v_ind_coeff = calc_circulation(inlet_condition.V_app_infs, sail_set.panels1d)

        df_gamma = pd.DataFrame({'gamma_magnitute': gamma_magnitude})
        df_gamma.to_csv(f'expected_gamma_magnitute_{suffix}.csv', index=False)
        expected_df_gamma = pd.read_csv(os.path.join(case_dir, f'expected_gamma_magnitute_{suffix}.csv'))
        assert_frame_equal(df_gamma, expected_df_gamma)

        V_induced_at_ctrl_p, V_app_fs_at_ctrl_p = calculate_app_fs(inlet_condition, v_ind_coeff, gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, sail_set.panels1d)
        # to calculate forces one have to recalculate induceed wind at cp (centre of pressure), not at ctr_point!
        V_app_fs_at_cp, V_induced_at_cp = calc_V_at_cp(inlet_condition.V_app_infs, gamma_magnitude, sail_set.panels1d)
        inviscid_flow_results_llt = prepare_inviscid_flow_results_llt(
            V_app_fs_at_cp, V_induced_at_cp, gamma_magnitude,
            sail_set, inlet_condition,
            self.csys_transformations)

        self._check_df_results(suffix, inviscid_flow_results_llt, inlet_condition, sail_set)

    def test_calc_forces_and_moments_use_vlm_as_llt(self):
        suffix = 'llt'
        n_chordwise = 1

        sail_set, inlet_condition = self._prepare_sail_set(n_spanwise=10, n_chordwise=n_chordwise)
        gamma_magnitude, v_ind_coeff = calc_circulation(inlet_condition.V_app_infs, sail_set.panels1d)

        df_gamma = pd.DataFrame({'gamma_magnitute': gamma_magnitude})
        df_gamma.to_csv(f'expected_gamma_magnitute_{suffix}.csv', index=False)
        expected_df_gamma = pd.read_csv(os.path.join(case_dir, f'expected_gamma_magnitute_{suffix}.csv'))
        assert_frame_equal(df_gamma, expected_df_gamma)

        V_induced_at_ctrl_p, V_app_fs_at_ctrl_p = calculate_app_fs(inlet_condition, v_ind_coeff, gamma_magnitude)
        assert is_no_flux_BC_satisfied(V_app_fs_at_ctrl_p, sail_set.panels1d)
        inviscid_flow_results_vlm = prepare_inviscid_flow_results_vlm(gamma_magnitude,
                                                                   sail_set, inlet_condition,
                                                                   self.csys_transformations)

        self._check_df_results(suffix, inviscid_flow_results_vlm, inlet_condition, sail_set)
