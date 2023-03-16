import numpy as np
from unittest import TestCase

import sailing_vlm.runner.aircraft as ac
from sailing_vlm.rotations.csys_transformations import CSYS_transformations
from sailing_vlm.runner.sail import Wind, Sail
from sailing_vlm.solver.vlm import Vlm
from sailing_vlm.solver.coefs import get_vlm_Cxyz


class TestValidation(TestCase):
    
#     >>> 
# >>> 
# >>> )
# >>> a.get_Cxyz()
# (0.023472780216173314, 0.8546846987984326, 0.0, array([0.14377078, 5.23494378, 0.        ]), array([1., 0., 0.]), 10.0, 6.125)
# ```
# SAIL
# 0.023472780216173342 0.8546846987984335 6.996235308182204e-34 [1.43770779e-01 5.23494378e+00 4.28519413e-33] [1. 0. 0.] 10.0 6.125
#     
    def test_flat_plate(self):
        # AIRCRAFT
        chord_length = 1.0
        half_span_length = 5.0
        AoA = 10.0
        n_spanwise = 32
        n_chordwise = 8
        V = np.array([1.0, 0., 0.])
        rho = 1.225
        gamma_orientation = -1.0
        
        a = ac.Aircraft(chord_length, half_span_length, AoA, n_spanwise, n_chordwise, V, rho, gamma_orientation)
        aircraft_Cxyz = a.Cxyz           
        # SAIL
    
        # SOLVER SETTINGS
        n_spanwise = 16  # No of control points (above the water) per sail, recommended: 50
        n_chordwise = 8 # No of control points (above the water) per sail, recommended: 50
        interpolation_type = "linear"  # either "spline" or "linear"
        LLT_twist = "real_twist"  # defines how the Lifting Line discretize the sail twist.
        # It can be "sheeting_angle_const" or "average_const" or "real_twist"

        # SAILING CONDITIONS
        leeway_deg = 0.    # [deg]
        heel_deg = 0.     # [deg]
        SOG_yacht = 0.   # [m/s] yacht speed - speed over ground (leeway is a separate variable)
        tws_ref = 1.     # [m/s] true wind speed
        alpha_true_wind_deg = 0.   # [deg] true wind angle (with reference to course over ground) => Course Wind Angle to the boat track = true wind angle to centerline + Leeway
        reference_water_level_for_wind_profile = 0.  # [m] this is an attempt to mimick the deck effect
        wind_exp_coeff = 0.  # [-] coefficient to determine the exponential wind profile
        wind_reference_measurment_height = 10.  # [m] reference height for exponential wind profile
        rho = 1.225  # air density [kg/m3]
        wind_profile = 'flat' # allowed: 'exponential' or 'flat' or 'logarithmic'
        roughness = 0.05 # for logarithmic profile only

        # GEOMETRY OF THE RIG
        main_sail_luff = 5.0 #10. # 12.4  # [m]
        jib_luff = 10.0  # [m]
        foretriangle_height = 11.50  # [m]
        foretriangle_base = 3.90  # [m]
        sheer_above_waterline = 0.  # [m]
        boom_above_sheer = 0. #1.3  # [m]
        rake_deg = 90.  # rake angle [deg]
        mast_LOA = 0.0  # [m]

        # INPUT - GEOMETRY OF THE SAIL
        sails_def = 'main' # definition of sail set, possible: 'jib' or 'main' or 'jib_and_main'
        main_sail_girths = np.array([0.00, 1./8, 1./4, 1./2, 3./4, 7./8, 1.00])
        main_sail_chords = np.array([1.00]* len(main_sail_girths)) # np.array([4.00, 3.82, 3.64, 3.20, 2.64, 2.32, 2.00])
        main_sail_centerline_twist_deg = -10+ 0* main_sail_girths # 10 + 12. * main_sail_girths  #

        # First digit describing maximum camber as percentage of the chord.
        # Second digit describing the distance of maximum camber from the airfoil leading edge in tenths of the chord.
        jib_sail_camber= 0*np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        jib_sail_camber_distance_from_luff = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) # starting from leading edge
        main_sail_camber= 0*np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        main_sail_camber_distance_from_luff = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # starting from leading edge

        jib_girths = np.array([0.00, 1./4, 1./2, 3./4, 1.00])
        jib_chords = 0*np.array([3.80, 2.98, 2.15, 1.33, 0.5]) - 0*0.4
        jib_centerline_twist_deg =  0*(10+5)  + 0*15. * jib_girths # (10+5)  + 15. * jib_girths # 

        reference_level_for_moments = np.array([0, 0, 0])  # [yaw_reference, sway_reference, heeling_reference]

        # GEOMETRY OF THE KEEL
        center_of_lateral_resistance_upright = np.array([0, 0, -1.0])  # [m] the coordinates for a yacht standing in upright position

        
        csys_transformations = CSYS_transformations(
        heel_deg, leeway_deg,
        v_from_original_xyz_2_reference_csys_xyz=reference_level_for_moments)

        w = Wind(alpha_true_wind_deg, tws_ref,SOG_yacht, wind_exp_coeff, wind_reference_measurment_height, reference_water_level_for_wind_profile, roughness, wind_profile)
        w_profile = w.profile

        s = Sail(n_spanwise, n_chordwise, csys_transformations, sheer_above_waterline,
                rake_deg, boom_above_sheer, mast_LOA,
                main_sail_luff, main_sail_girths, main_sail_chords, main_sail_centerline_twist_deg, main_sail_camber,main_sail_camber_distance_from_luff,
                foretriangle_base, foretriangle_height, 
                jib_luff, jib_girths, jib_chords, jib_centerline_twist_deg, jib_sail_camber, jib_sail_camber_distance_from_luff,
                sails_def, LLT_twist, interpolation_type)
        sail_set = s.sail_set
        myvlm = Vlm(sail_set.panels, n_chordwise, n_spanwise, rho, w_profile, sail_set.trailing_edge_info, sail_set.leading_edge_info)
        
        
        S = 2*main_sail_luff * main_sail_chords[0]
        sail_Cxyz = get_vlm_Cxyz(myvlm.force, np.array(w_profile.get_true_wind_speed_at_h(1.0)), rho, S)
        
        # print(f'{Cxyz[0]} vs {sail_Cxyz[0]}')
        # print(f'{Cxyz[1]} vs {sail_Cxyz[1]}')
        # print(f'{Cxyz[2]} vs {sail_Cxyz[2]}')
        np.testing.assert_almost_equal(aircraft_Cxyz[0], sail_Cxyz[0])
        np.testing.assert_almost_equal(aircraft_Cxyz[1], sail_Cxyz[1])
        np.testing.assert_almost_equal(aircraft_Cxyz[2], sail_Cxyz[2])