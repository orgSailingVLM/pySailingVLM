import shutil

from pySailingVLM.yacht_geometry.sail_factory import SailFactory
from pySailingVLM.yacht_geometry.sail_geometry import SailSet
from pySailingVLM.rotations.csys_transformations import CSYS_transformations
from pySailingVLM.solver.interpolator import Interpolator
from pySailingVLM.yacht_geometry.hull_geometry import HullGeometry
from pySailingVLM.inlet.winds import ExpWindProfile
from pySailingVLM.results.inviscid_flow import InviscidFlowResults

from pySailingVLM.results.save_utils import save_results_to_file

from pySailingVLM.solver.vlm import Vlm
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest import TestCase

from pySailingVLM.tests.input_files.case_data_for_vlm_runner import *

class TestJibRunner(TestCase):
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
            LLT_twist=LLT_twist,
            interpolated_camber=self.interpolator.interpolate_girths(jib_girths, jib_camber, n_spanwise + 1),
            interpolated_distance_from_luff=self.interpolator.interpolate_girths(jib_girths, jib_distance_from_luff, n_spanwise + 1),
            )

        main_sail_geometry = sail_factory.make_main_sail(
            main_sail_luff=main_sail_luff,
            boom_above_sheer=boom_above_sheer,
            main_sail_chords=self.interpolator.interpolate_girths(main_sail_girths, main_sail_chords, n_spanwise + 1),
            sail_twist_deg=self.interpolator.interpolate_girths(main_sail_girths, main_sail_centerline_twist_deg, n_spanwise + 1),
            LLT_twist=LLT_twist,
            interpolated_camber=self.interpolator.interpolate_girths(main_sail_girths, main_sail_camber, n_spanwise + 1),
            interpolated_distance_from_luff=self.interpolator.interpolate_girths(main_sail_girths, main_sail_distance_from_luff, n_spanwise + 1)
            )

        sail_set = SailSet([jib_geometry, main_sail_geometry])

        myvlm = Vlm(sail_set.panels, n_chordwise, n_spanwise, rho, self.wind, sail_set.trailing_edge_info, sail_set.leading_edge_info)

        inviscid_flow_results = InviscidFlowResults(sail_set, self.csys_transformations, myvlm)
        return sail_set, myvlm, inviscid_flow_results

   
    def test_calc_forces_and_moments_vlm(self):
        nc = 2
        ns = 4

        sail_set, myvlm, inviscid_flow_results = self._prepare_sail_set(n_spanwise=ns, n_chordwise=nc)

        res_dir = 'dummy_file'
        df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, 
            self.csys_transformations, inviscid_flow_results, sail_set, res_dir)
        ##
        df_gamma = pd.DataFrame({'gamma_magnitute': myvlm.gamma_magnitude})

        expected_df_gamma = pd.read_csv(os.path.join(case_dir, 'expected_gamma_magnitute_vlm.csv'))
        df_components_expected = pd.read_csv(os.path.join(case_dir, 'expected_components_vlm.csv'), index_col=0)
        df_inlet_IC_expected = pd.read_csv(os.path.join(case_dir, 'expected_inlet_IC_vlm.csv'), index_col=0)
        df_integrals_expected = pd.read_csv(os.path.join(case_dir, 'expected_integrals_vlm.csv'), index_col=0)
        
        pd.testing.assert_frame_equal(df_components, df_components_expected)
        pd.testing.assert_frame_equal(df_integrals, df_integrals_expected)
        pd.testing.assert_frame_equal(df_inlet_IC, df_inlet_IC_expected)

        pd.testing.assert_frame_equal(df_gamma, expected_df_gamma)
        shutil.rmtree(res_dir)
