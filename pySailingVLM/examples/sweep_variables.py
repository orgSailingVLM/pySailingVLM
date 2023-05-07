
import os
import numpy as np
import time

# OUTPUT DIR
output_dir = {
    'case_name':  os.path.basename(__file__),  # get name of the current file
    'case_dir' : os.path.dirname(os.path.realpath(__file__)),  # get dir of the current file
    'name': os.path.join("results_example_jib_and_mainsail_vlm", time.strftime("%Y-%m-%d_%Hh%Mm%Ss")),
    'file_name': 'my_fancy_results', # name of xlsx excel file
}

# SOLVER SETTINGS
solver = {
    'n_spanwise':  4,  # No of control points (above the water) per sail, recommended: 50
    'n_chordwise': 1, # No of control points (above the water) per sail, recommended: 50
    'interpolation_type': "linear",  # either "spline" or "linear"
    'LLT_twist': "real_twist",  # defines how the Lifting Line discretize the sail twist.
    # It can be "sheeting_angle_const" or "average_const" or "real_twist"
}

# SAILING CONDITIONS
conditions = {
    'leeway_deg': 0.,    # [deg]
    'heel_deg': 0.,     # [deg]
    'SOG_yacht': 0.,   # [m/s] yacht speed - speed over ground (leeway is a separate variable)
    'tws_ref': 1.,     # [m/s] true wind speed
    'alpha_true_wind_deg': 0.,   # [deg] true wind angle (with reference to course over ground) => Course Wind Angle to the boat track = true wind angle to centerline + Leeway
    'reference_water_level_for_wind_profile': -0.,  # [m] this is an attempt to mimick the deck effect
    # by lowering the sheer_above_waterline
    # while keeping the wind profile as in original geometry
    # this shall be negative (H = sail_ctrl_point - water_level)
    'wind_exp_coeff': 0.,  # [-] coefficient to determine the exponential wind profile
    'wind_reference_measurment_height': 10.,  # [m] reference height for exponential wind profile
    'rho': 1.225,  # air density [kg/m3]
    'wind_profile': 'flat', # allowed: 'exponential' or 'flat' or 'logarithmic'
    'roughness': 0.05, # for logarithmic profile only 
}


# GEOMETRY OF THE RIG
rig = {
    'main_sail_luff': 0.5,  # [m]
    'jib_luff': 10.0,  # [m]
    'foretriangle_height': 11.50,  # [m]
    'foretriangle_base': 3.90,  # [m]
    'sheer_above_waterline': 0.,#[m]
    'boom_above_sheer': 0., # [m],
    'rake_deg': 90 + 45.,  # rake angle [deg]
    'mast_LOA': 0.,  # [m]
    'sails_def': 'main', # definition of sail set, possible: 'jib' or 'main' or 'jib_and_main'
    'main_sail_girths': np.array([0.00, 1./8, 1./4, 1./2, 3./4, 7./8, 1.00]),
    'jib_girths': np.array([0.00, 1./4, 1./2, 3./4, 1.00]),
}

# INPUT - GEOMETRY OF THE SAIL
# INFO for camber:
# First digit describing maximum camber as percentage of the chord.
# Second digit describing the distance of maximum camber from the airfoil leading edge in tenths of the chord.

main_sail = {
    'main_sail_chords': np.array([0.2]* len(rig['main_sail_girths'])),
    'main_sail_centerline_twist_deg': -2 + 0* rig['main_sail_girths'] , 
    'main_sail_camber': 0*np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    'main_sail_camber_distance_from_luff': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), # starting from leading edge
}


jib_sail = {
    'jib_chords': 0* np.array([3.80, 2.98, 2.15, 1.33, 0.5]),
    'jib_centerline_twist_deg': 0*(10+5)  + 0*15. * rig['jib_girths'], 
    'jib_sail_camber': 0*np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
    'jib_sail_camber_distance_from_luff': np.array([0.5, 0.5, 0.5, 0.5, 0.5]), # starting from leading edge   
}

# REFERENCE CSYS
# The origin of the default CSYS is located @ waterline level and aft face of the mast
# The positive x-coord: towards stern
# The positive y-coord: towards leeward side
# The positive z-coord: above the water
# To shift the default CSYS, adjust the 'reference_level_for_moments' variable.
# Shifted CSYS = original + reference_level_for_moments
# As a results the moments will be calculated around the new origin.

# yaw_reference [m] - distance from the aft of the mast towards stern, at which the yawing moment is calculated.
# sway_reference [m] - distance from the aft of the mast towards leeward side. 0 for symmetric yachts ;)
# heeling_reference [m] - distance from the water level,  at which the heeling moment is calculated.
csys = {
    'reference_level_for_moments': np.array([0, 0, 0]),  # [yaw_reference, sway_reference, heeling_reference]
}

# GEOMETRY OF THE KEEL
# to estimate heeling moment from keel, does not influence the optimizer.
# reminder: the z coord shall be negative (under the water)
keel = {
    'center_of_lateral_resistance_upright': np.array([0, 0, -1.0]),  # [m] the coordinates for a yacht standing in upright position
}