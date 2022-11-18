import shutil

from sailing_vlm.yacht_geometry.sail_factory import SailFactory
from sailing_vlm.yacht_geometry.sail_geometry import SailSet
from sailing_vlm.rotations.csys_transformations import CSYS_transformations
from sailing_vlm.solver.interpolator import Interpolator
from sailing_vlm.yacht_geometry.hull_geometry import HullGeometry
from sailing_vlm.inlet.inlet_conditions import InletConditions
from sailing_vlm.inlet.winds import ExpWindProfile

from sailing_vlm.solver.forces import calc_V_at_cp
from sailing_vlm.results.save_utils import save_results_to_file
from sailing_vlm.solver.panels_plotter import display_panels_xyz_and_winds
#from sailing_vlm.solver.vlm_solver import is_no_flux_BC_satisfied

#from sailing_vlm.solver.vlm_solver import calc_circulation
from sailing_vlm.results.inviscid_flow import  prepare_inviscid_flow_results_vlm #, prepare_inviscid_flow_results_llt,
#from sailing_vlm.solver.vlm_solver import calculate_app_fs

from sailing_vlm.solver.vlm import Vlm
import pandas as pd
from pandas.util.testing import assert_frame_equal
from unittest import TestCase


from sailing_vlm.tests.input_files.case_data_for_vlm_runner import *
# np.set_printoptions(precision=3, suppress=True)


class TestVLM_solver(TestCase):
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
        #inlet_condition = InletConditions(self.wind, rho=rho, panels1D=sail_set.panels1d)
        
        myvlm = Vlm(sail_set.panels, n_chordwise, n_spanwise, rho, self.wind, sail_set.trailing_edge_info, sail_set.leading_edge_info)

        return sail_set, myvlm# , inlet_condition




    def _check_df_results(self, suffix, myvlm, csys_transformations, inviscid_flow_results, sail_set):
        inviscid_flow_results.estimate_heeling_moment_from_keel(self.hull.center_of_lateral_resistance)
        # display_panels_xyz_and_winds(self.sail_set.panels1d, self.inlet_condition, inviscid_flow_results, self.hull, show_plot=False)

                             
        df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, 
            csys_transformations, inviscid_flow_results, sail_set, output_dir_name)
        shutil.copy(os.path.join(case_dir, case_name), os.path.join(output_dir_name, case_name))


        df_components.to_csv(f'expected_df_components_{suffix}.csv')
        df_integrals.to_csv(f'expected_df_integrals_{suffix}.csv', index=False)
        df_inlet_IC.to_csv(f'expected_df_inlet_IC_{suffix}.csv')

        expected_df_integrals = pd.read_csv(os.path.join(case_dir, f'expected_df_integrals_{suffix}.csv'))
        assert_frame_equal(df_integrals, expected_df_integrals)

        expected_df_inlet_ic = pd.read_csv(os.path.join(case_dir, f'expected_df_inlet_IC_{suffix}.csv'))
        assert_frame_equal(df_inlet_IC, expected_df_inlet_ic)

        expected_df_components = pd.read_csv(os.path.join(case_dir, f'expected_df_components_{suffix}.csv'))
        assert_frame_equal(df_components, expected_df_components)

    def test_calc_forces_and_moments_vlm(self):
        suffix = 'vlm'
        n_chordwise = 3

        sail_set, myvlm = self._prepare_sail_set(n_spanwise=2, n_chordwise=n_chordwise)

        df_gamma = pd.DataFrame({'gamma_magnitute': myvlm.gamma_magnitude})
        df_gamma.to_csv(f'expected_gamma_magnitute_{suffix}.csv', index=False)
        expected_df_gamma = pd.read_csv(os.path.join(case_dir, f'expected_gamma_magnitute_{suffix}.csv'))
        np.testing.assert_almost_equal(df_gamma['gamma_magnitute'].to_numpy(), expected_df_gamma['gamma_magnitute'].to_numpy())

        inviscid_flow_results_vlm = prepare_inviscid_flow_results_vlm(sail_set, self.csys_transformations, myvlm)
        self._check_df_results(suffix, myvlm, self.csys_transformations, inviscid_flow_results_vlm, sail_set)
