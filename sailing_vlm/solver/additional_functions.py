import numpy as np
import pandas as pd
from typing import List, Tuple
from numpy.linalg import norm

import numba

#@numba.jit(numba.float64[::1](numba.float64[::1]), nopython=True, debug=False)
def normalize(x):
    # xn = x / norm(x)
    xn = x / np.linalg.norm(x)
    return xn

def extract_above_water_quantities(quantities, cp_points):
    
    # for jib and main, quantities is always dividable by 2
    half = int(quantities.shape[0] / 2)
    above_water_quantities = quantities[0:half]
    total_above_water_quantities = np.sum(above_water_quantities, axis=0)  # heeling, sway, yaw (z-axis)
    return above_water_quantities, total_above_water_quantities


# circular import
#  csys_transformations: CSYS_transformations
def get_cp_strainght_yacht(cp_points : np.ndarray, csys_transformations) -> np.ndarray:
    """
    cp_strainght_yacht get center of pressure points straight to bridge

    :param np.ndarray cp_points: center of pressure points
    :return np.ndarray: staright center of pressure points
    """
    return np.array([csys_transformations.reverse_rotations_with_mirror(p) for p in cp_points])

def cp_to_girths(sail_cp_straight_yacht, tack_mounting):
    sail_cp_straight_yacht_z = sail_cp_straight_yacht[:, 2]
    z_as_girths = (sail_cp_straight_yacht_z - tack_mounting[2]) / (max(sail_cp_straight_yacht_z) - tack_mounting[2])
    return z_as_girths

# circular import
#  csys_transformations: CSYS_transformations
# sail_set : SailSet problem with import
def get_cp_z_as_girths_all(sail_set, csys_transformations, center_of_pressure : np.ndarray) -> np.ndarray: 
    """
    cp_z_as_girths_all get center of pressure staright z as girths

    :param SailSet sail_set: Sail set object
    :param CSYS_transformations csys_transformations: csys transformations
    :param np.ndarray center_of_pressure: array with all center of pressure points (for all sails and for above and under water)

    :return np.ndarray: y_as_girths for all sails (with above and under water points)
    """
    n = len(sail_set.sails)
    # 2* bo mamy odpicie lustrzane 
    chunks_cp_points = np.array_split(center_of_pressure, 2 * n)

    nxm = chunks_cp_points[0].shape[0]
    cp_z_as_girths_all = np.array([])
    #cp_straight_yacht_all = np.array([])
    
    cp_straight_yacht_all = np.empty((0,3))
    for i in range(n):
        sail_cp_points = np.concatenate([chunks_cp_points[i], chunks_cp_points[i+n]])
        sail_cp_straight_yacht = get_cp_strainght_yacht(sail_cp_points, csys_transformations)
        z_as_girths = cp_to_girths(sail_cp_straight_yacht, sail_set.sails[i].tack_mounting)
        cp_z_as_girths_all = np.append(cp_z_as_girths_all, z_as_girths)
        #cp_straight_yacht_all = np.append(cp_straight_yacht_all, sail_cp_straight_yacht)
        cp_straight_yacht_all = np.append(cp_straight_yacht_all, sail_cp_straight_yacht, axis=0)

    # ulozenie w array cp_z_as_girths_all :
    # jib above, jib under, main abobe, main under
    return cp_z_as_girths_all, cp_straight_yacht_all


# to policzyc dla wszystkich cpsow
# potem wziac gorna czesc czyli pierwsza polowe by miec nad woda
# import problem
# sail set  sail_set : SailSet
def get_cp_z_as_girths_all_above(cp_z_as_girths_all : np.ndarray, sail_set):
    # 2 bo mamy odbicie 
    # parzyste numery to sa te nad woda
    n = 2 * len(sail_set.sails)
    z_splitted = np.split(cp_z_as_girths_all, n)
    # [::2] gets every second element from array z_splitted
    cp_z_as_girths_all_above = np.asarray(z_splitted[::2]).flatten()
    # cp_z_as_girths_all_above size
    repeat = int(cp_z_as_girths_all_above.shape[0] / 2)
    names = np.array([])
    
    for sail in sail_set.sails:
        rep_names = np.repeat(sail.name, repeat)
        names = np.append(names, rep_names)
      
    return cp_z_as_girths_all_above, names