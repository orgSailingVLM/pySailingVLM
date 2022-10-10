
import os
import numpy as np
import time

# OUTPUT DIR
case_name = os.path.basename(__file__)  # get name of the current file
case_dir = os.path.dirname(os.path.realpath(__file__))  # get dir of the current file
time_stamp = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
output_dir_name = os.path.join("results_example_jib_and_mainsail_vlm", time_stamp)

# SOLVER SETTINGS
n_spanwise = 15  # No of control points (above the water) per sail, recommended: 50
n_chordwise = 5  # No of control points (above the water) per sail, recommended: 50
interpolation_type = "spline"  # either "spline" or "linear"
LLT_twist = "real_twist"  # defines how the Lifting Line discretize the sail twist.
# It can be "sheeting_angle_const" or "average_const" or "real_twist"

# SAILING CONDITIONS
leeway_deg = 5.    # [deg]
heel_deg = 10.     # [deg]
SOG_yacht = 4.63   # [m/s] yacht speed - speed over ground (leeway is a separate variable)
tws_ref = 4.63     # [m/s] true wind speed
alpha_true_wind_deg = 50.   # [deg] true wind angle (with reference to course over ground) => Course Wind Angle to the boat track = true wind angle to centerline + Leeway
reference_water_level_for_wind_profile = -0.  # [m] this is an attempt to mimick the deck effect
# by lowering the sheer_above_waterline
# while keeping the wind profile as in original geometry
# this shall be negative (H = sail_ctrl_point - water_level)
wind_exp_coeff = 0.1428  # [-] coefficient to determine the exponential wind profile
wind_reference_measurment_height = 10.  # [m] reference height for exponential wind profile
rho = 1.225  # air density [kg/m3]

# GEOMETRY OF THE RIG
main_sail_luff = 12.4  # [m]
jib_luff = 10.0  # [m]
foretriangle_height = 11.50  # [m]
foretriangle_base = 3.90  # [m]
sheer_above_waterline = 1.20  # [m]
boom_above_sheer = 1.30  # [m]
rake_deg = 92.  # rake angle [deg]
mast_LOA = 0.15  # [m]

# INPUT - GEOMETRY OF THE SAIL
main_sail_girths = np.array([0.00, 1./8, 1./4, 1./2, 3./4, 7./8, 1.00])
main_sail_chords = np.array([4.00, 3.82, 3.64, 3.20, 2.64, 2.32, 2.00])
main_sail_centerline_twist_deg = 12. * main_sail_girths + 7

jib_girths = np.array([0.00, 1./4, 1./2, 3./4, 1.00])
jib_chords = np.array([3.80, 2.98, 2.15, 1.33, 0.5]) - 0.4
jib_centerline_twist_deg = 15. * jib_girths + 5

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
reference_level_for_moments = np.array([0, 0, 0])  # [yaw_reference, sway_reference, heeling_reference]

# GEOMETRY OF THE KEEL
# to estimate heeling moment from keel, does not influence the optimizer.
# reminder: the z coord shall be negative (under the water)
center_of_lateral_resistance_upright = np.array([0, 0, -1.0])  # [m] the coordinates for a yacht standing in upright position
