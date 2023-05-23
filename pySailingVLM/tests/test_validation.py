import numpy as np
from unittest import TestCase

import pySailingVLM.runner.aircraft as ac
from pySailingVLM.rotations.csys_transformations import CSYS_transformations
from pySailingVLM.runner.sail import Wind, Sail
from pySailingVLM.solver.vlm import Vlm
from pySailingVLM.results.inviscid_flow import InviscidFlowResults
from pySailingVLM.solver.coefs import get_vlm_Cxyz
from pySailingVLM.rotations.geometry_calc import rotation_matrix
from pySailingVLM.results.inviscid_flow import InviscidFlowResults

from pySailingVLM.runner.container import Rig, Conditions, Solver, MainSail, JibSail
class TestValidation(TestCase):
    def calculate_main_sail(self, rho: float, tws_ref: float, chord_length: float, half_wing_span:  float, AoA_deg: float, sweep_angle_deg : float, n_spanwise: int, n_chordwise: int):
        
        solver_args = {
            'n_spanwise':  n_spanwise,  # No of control points (above the water) per sail, recommended: 50
            'n_chordwise': n_chordwise, # No of control points (above the water) per sail, recommended: 50
            'interpolation_type': "linear",  # either "spline" or "linear"
            'LLT_twist': "real_twist",  # defines how the Lifting Line discretize the sail twist.
        }
        
        conditions_args = {
            'leeway_deg': 0.,    # [deg]
            'heel_deg': 0.,     # [deg]
            'SOG_yacht': 0.,   # [m/s] yacht speed - speed over ground (leeway is a separate variable)
            'tws_ref': tws_ref,     # [m/s] true wind speed
            'alpha_true_wind_deg': AoA_deg,   # [deg] true wind angle (with reference to course over ground) => Course Wind Angle to the boat track = true wind angle to centerline + Leeway
            'reference_water_level_for_wind_profile': -0.,  # [m] this is an attempt to mimick the deck effect
            # by lowering the sheer_above_waterline
            # while keeping the wind profile as in original geometry
            # this shall be negative (H = sail_ctrl_point - water_level)
            'wind_exp_coeff': 0.,  # [-] coefficient to determine the exponential wind profile
            'wind_reference_measurment_height': 10.,  # [m] reference height for exponential wind profile
            'rho': rho,  # air density [kg/m3]
            'wind_profile': 'flat', # allowed: 'exponential' or 'flat' or 'logarithmic'
            'roughness': 0.05, # for logarithmic profile only 
        }

        rig_args = {
            'main_sail_luff': half_wing_span / np.cos(np.deg2rad(sweep_angle_deg)),  # [m]
            'jib_luff': 10.0,  # [m]
            'foretriangle_height': 11.50,  # [m]
            'foretriangle_base': 3.90,  # [m]
            'sheer_above_waterline': 0.,#[m]
            'boom_above_sheer': 0., # [m],
            'rake_deg': 90. + sweep_angle_deg,  # rake angle [deg]
            'mast_LOA': 0.,  # [m]
            'sails_def': 'main', # definition of sail set, possible: 'jib' or 'main' or 'jib_and_main'
        }

        mgirths =  np.array([0.00, 1./8, 1./4, 1./2, 3./4, 7./8, 1.00])
        mchords = np.array([chord_length]* len(mgirths))

  
        main_args = {
            'girths' : mgirths,
            'chords': mchords,
            'centerline_twist_deg': 0*mgirths,
            'camber': 0*np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
            'camber_distance_from_luff': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        }

        solver =  Solver(**solver_args)
        conditions = Conditions(**conditions_args) 
        rig = Rig(**rig_args)
        main = MainSail(**main_args)

        reference_level_for_moments = np.array([0, 0, 0])
        csys_transformations = CSYS_transformations(conditions.heel_deg, conditions.leeway_deg, v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

        w = Wind(conditions)
        s = Sail(solver, rig, main, None, csys_transformations)
        sail_set = s.sail_set
        myvlm = Vlm(sail_set.panels, solver.n_chordwise, solver.n_spanwise, conditions.rho, w.profile, sail_set.trailing_edge_info, sail_set.leading_edge_info)

        inviscid_flow_results = InviscidFlowResults(sail_set, csys_transformations, myvlm)
      
        A = rotation_matrix([0, 0, 1], np.deg2rad(-AoA_deg))
        sail_mirror_multiplier = 2 # because inviscid_flow_results.F_xyz_total gets total only above water # it is the same as  total of myvlm.force
        F_xyz_total = sail_mirror_multiplier*inviscid_flow_results.F_xyz_total
        F = np.dot(A, F_xyz_total)  # By definition, the lift force is perpendicular to V_inf

        S = 2 * half_wing_span * chord_length
        q = 0.5 * rho * (np.linalg.norm(tws_ref) ** 2) * S
        sail_Cxyz = F / q

        inviscid_flow_results = InviscidFlowResults(sail_set, csys_transformations, myvlm)
        
        return sail_Cxyz, F, myvlm, inviscid_flow_results

    def test_sweep_wing(self):
        # page 365, Example 7.2 Aerodynamics for engineers, John J.Bertin
        # https://airloads.net/Downloads/Textbooks/Aerodynamics-for-engineers-%20John%20J.Bertin.pdf
        
        chord_length = 0.2             
        half_wing_span = 0.5 
        AoA_deg = 4.
        sweep_angle_deg = 45.
        tws_ref = 1.0
        rho = 1.0
        # SOLVER SETTINGS
        n_spanwise = 4  # No of control points (above the water) per sail, recommended: 50
        n_chordwise = 1 # No of control points (above the water) per sail, recommended: 50
        
        sail_Cxyz, F, myvlm, _ = self.calculate_main_sail(rho, tws_ref, chord_length, half_wing_span, AoA_deg, sweep_angle_deg, n_spanwise, n_chordwise)
        
        a_VLM = sail_Cxyz[1] / np.deg2rad(AoA_deg)
        np.testing.assert_almost_equal(myvlm.rings[3,1], np.array([0.425, 0., 0.375]), decimal=3)
        np.testing.assert_almost_equal(myvlm.rings[3, 2], np.array([0.550, 0., 0.500]), decimal=3)
        np.testing.assert_almost_equal(myvlm.ctr_p[3], np.array([0.5875, 0., 0.4375]), decimal=4)
        np.testing.assert_almost_equal(myvlm.gamma_magnitude[0:4], np.array([0.0239377, 0.02521229, 0.02511784, 0.02187719]), decimal=4)
        np.testing.assert_almost_equal(F, np.array([0.00031786, 0.02402581, 0.00135887]), decimal=5)
        np.testing.assert_almost_equal(sail_Cxyz[1],0.240258, decimal=4)
        np.testing.assert_almost_equal(a_VLM, 3.441443, decimal=2)
        
        
        gamma_ref = 4*np.pi*2*half_wing_span*tws_ref*np.deg2rad(AoA_deg)*np.array([0.0273, 0.0287, 0.0286, 0.0250]) #get_gamma_reference(2*half_wing_span, tws_ref, AoA_deg) # Eq 7.48
        dy = 2*half_wing_span/(2*len(gamma_ref))
        L_ref = 2*rho*tws_ref*tws_ref*np.sum(gamma_ref*dy)    #eq 7.50b
        
        S = 2 * half_wing_span * chord_length
        q = 0.5 * rho * (np.linalg.norm(tws_ref) ** 2) * S
        
        CL_ref = L_ref / q
        a_ref = CL_ref / np.deg2rad(AoA_deg)

        L_ref2 = rho * tws_ref * tws_ref * 4 * half_wing_span * half_wing_span * np.pi * np.deg2rad(AoA_deg) * 0.1096
        CL_ref2 = 1.096*np.pi*np.deg2rad(AoA_deg)

        np.testing.assert_almost_equal(L_ref, L_ref2, decimal=6)
        np.testing.assert_almost_equal(L_ref, F[1], decimal=4)
        np.testing.assert_almost_equal(CL_ref, CL_ref2, decimal=6)
        np.testing.assert_almost_equal(CL_ref, 0.240379, decimal=6)
        np.testing.assert_almost_equal(a_ref, 3.443185, decimal=6)

    def test_flat_plate(self):
        # comparison between theory (aircraft) and sailing vlm approach
        # tested numerically with tornado in matlab
        
        chord_length = 1.0             
        half_wing_span = 5.0 
        AoA_deg = 10.
        sweep_angle_deg = 0.
        tws_ref = 1.0
        V = np.array([tws_ref, 0., 0.])
        rho = 1.225
        gamma_orientation = -1.0
        n_spanwise = 32  # No of control points (above the water) per sail, recommended: 50
        n_chordwise = 8 # No of control points (above the water) per sail, recommended: 50
        
        # AIRCRAFT
        a = ac.Aircraft(chord_length, half_wing_span, AoA_deg, n_spanwise, n_chordwise, V, rho, gamma_orientation)
        aircraft_Cxyz = a.Cxyz 

        # n_spanwise / 2 because SAIL has underwater part also
        sail_Cxyz, F, myvlm, inviscid_flow_results = self.calculate_main_sail(rho, tws_ref, chord_length, half_wing_span, AoA_deg, sweep_angle_deg, int(n_spanwise / 2), n_chordwise)
        
        np.testing.assert_almost_equal(aircraft_Cxyz[0], sail_Cxyz[0])
        np.testing.assert_almost_equal(aircraft_Cxyz[1], sail_Cxyz[1])
        np.testing.assert_almost_equal(aircraft_Cxyz[2], sail_Cxyz[2])
        
        M_xyz_expected = np.array([ 7.16657636e-17, -2.77147486e-16,  1.26635518e+00])
        
        np.testing.assert_almost_equal(inviscid_flow_results.M_xyz, inviscid_flow_results.M_centerline_csys)
        np.testing.assert_almost_equal(np.sum(inviscid_flow_results.M_xyz, axis=0), M_xyz_expected)
        np.testing.assert_almost_equal(inviscid_flow_results.F_xyz, inviscid_flow_results.F_centerline)
        np.testing.assert_almost_equal(inviscid_flow_results.M_total_above_water_in_centerline_csys, np.sum(np.array_split(inviscid_flow_results.M_xyz, 2)[0], axis=0))
        np.testing.assert_almost_equal(inviscid_flow_results.M_xyz, inviscid_flow_results.M_centerline_csys)

        