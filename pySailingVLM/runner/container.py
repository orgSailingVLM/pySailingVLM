
# DO NOT DELETE ME !!!
# USED BY TYPE CHECKING IN DATACLASSES
import os
import numpy as np
import time
from dataclasses import dataclass, field
from typing import ClassVar
# OUTPUT DIR
@dataclass
class Output:
    case_name: str = os.path.basename(__file__)  # get name of the current file
    case_dir: str = os.path.dirname(os.path.realpath(__file__)) # get dir of the current file
    name: str = os.path.join("results_example_jib_and_mainsail_vlm", time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    file_name: str = 'my_fancy_results' # name of xlsx excel file

# SOLVER SETTINGS
@dataclass
class Solver:
    n_spanwise: int  = 4  # No of control points (above the water) per sail, recommended: 50
    n_chordwise: int = 4 # No of control points (above the water) per sail, recommended: 50
    interpolation_type: str = "spline"  # either "spline" or "linear"
    LLT_twist: ClassVar[str] = 'real_twist' # defines how the Lifting Line discretize the sail twist.
    # It can be "sheeting_angle_const" or "average_const" or "real_twist"


# SAILING CONDITIONS
@dataclass
class Conditions:
    leeway_deg: float = 5.    # [deg]
    heel_deg: float = 10.     # [deg]
    SOG_yacht: float = 4.63   # [m/s] yacht speed - speed over ground (leeway is a separate variable)
    tws_ref: float = 4.63     # [m/s] true wind speed
    alpha_true_wind_deg: float = 50.   # [deg] true wind angle (with reference to course over ground) => Course Wind Angle to the boat track = true wind angle to centerline + Leeway
    reference_water_level_for_wind_profile: float = -0.  # [m] this is an attempt to mimick the deck effect
    # by lowering the sheer_above_waterline
    # while keeping the wind profile as in original geometry
    # this shall be negative (H = sail_ctrl_point - water_level)
    wind_exp_coeff: float = 0.1428  # [-] coefficient to determine the exponential wind profile
    wind_reference_measurment_height: float = 10.  # [m] reference height for exponential wind profile
    rho: float = 1.225 # air density [kg/m3]
    wind_profile: str = 'exponential'# allowed: 'exponential' or 'flat' or 'logarithmic'
    roughness: float = 0.05 # for logarithmic profile only 



# GEOMETRY OF THE RIG
@dataclass
class Rig:
    main_sail_luff: float = 12.4  # [m]
    jib_luff: float = 10.0  # [m] 
    foretriangle_height: float = 11.50  # [m]
    foretriangle_base: float = 3.90  # [m]
    sheer_above_waterline: float = 1.2 #[m]
    boom_above_sheer: float = 1.3 # [m]
    rake_deg: float = 92.  # rake angle [deg]
    mast_LOA: float = 0.15  # [m]
    sails_def: float = 'jib_and_main' # definition of sail set, possible: 'jib' or 'main' or 'jib_and_main'


# INPUT - GEOMETRY OF THE SAIL
# INFO for camber:
# First digit describing maximum camber as percentage of the chord.
# Second digit describing the distance of maximum camber from the airfoil leading edge in tenths of the chord.

@dataclass
class MainSail:
    centerline_twist_deg: np.ndarray = None
    
    girths: np.ndarray = np.array([0.00, 1./8, 1./4, 1./2, 3./4, 7./8, 1.00])
    chords: np.ndarray = np.array([4.00, 3.82, 3.64, 3.20, 2.64, 2.32, 2.00])
    camber: np.ndarray = 5*np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    camber_distance_from_luff: np.ndarray = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # starting from leading edge
    
    def __post_init__(self):
        if self.centerline_twist_deg is None:
            self.centerline_twist_deg = 12. * self.girths + 5

@dataclass
class JibSail:
    centerline_twist_deg: np.ndarray = None
    
    girths: np.ndarray = np.array([0.00, 1./4, 1./2, 3./4, 1.00])
    chords: np.ndarray =np.array([3.80, 2.98, 2.15, 1.33, 0.5])
    camber: np.ndarray = 0*np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    camber_distance_from_luff: np.ndarray = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) # starting from leading edge   
    def __post_init__(self):
        if self.centerline_twist_deg is None:
            self.centerline_twist_deg = 15. * self.girths + 7

        
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
@dataclass
class Csys:
    reference_level_for_moments: np.ndarray =  np.array([0, 0, 0])  # [yaw_reference, sway_reference, heeling_reference]


# GEOMETRY OF THE KEEL
# to estimate heeling moment from keel, does not influence the optimizer.
# reminder: the z coord shall be negative (under the water)
@dataclass
class Keel:
    center_of_lateral_resistance_upright: np.ndarray =  np.array([0, 0, -1.0])  # [m] the coordinates for a yacht standing in upright position
