from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pySailingVLM.rotations.csys_transformations import CSYS_transformations
    from pySailingVLM.yacht_geometry.sail_geometry import SailSet

import numpy as np
import numba
import matplotlib.pyplot as plt
from typing import List, Tuple

from pySailingVLM.solver.velocity import vortex_ring, vortex_horseshoe




#@numba.jit(numba.float64[::1](numba.float64[::1]), nopython=True, debug=False)
def normalize(x):
    # xn = x / norm(x)
    xn = x / np.linalg.norm(x)
    return xn

def extract_above_water_quantities(quantities : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    extract_above_water_quantities get data above water

    :param np.ndarray quantities: data to be "searched" for above water quantities
    :return Tuple[np.ndarray, np.ndarray]: above water quantities, total above water quantities
    """
    
    # for jib and main, quantities is always dividable by 2
    half = int(quantities.shape[0] / 2)
    above_water_quantities = quantities[0:half]
    total_above_water_quantities = np.sum(above_water_quantities, axis=0)  # heeling, sway, yaw (z-axis)
    return above_water_quantities, total_above_water_quantities


# circular import
#  csys_transformations: CSYS_transformations
def get_cp_strainght_yacht(cp_points : np.ndarray, csys_transformations : CSYS_transformations) -> np.ndarray:
    """
    cp_strainght_yacht get center of pressure points straight to bridge

    :param np.ndarray cp_points: center of pressure points
    :param CSYS_transformations: : CSYS_transformations objetc
    :return np.ndarray: staright center of pressure points
    """
    return np.array([csys_transformations.reverse_rotations_with_mirror(p) for p in cp_points])

def cp_to_girths(sail_cp_straight_yacht : np.ndarray, tack_mounting : np.ndarray) -> np.ndarray:
    """
    cp_to_girths convert center of pressure points to girths

    :param np.ndarray sail_cp_straight_yacht: straighten yacht center of pressure points
    :param np.ndarray tack_mounting: tack mounting
    :return np.ndarray: center of pressure points as girths
    """
    
    sail_cp_straight_yacht_z = sail_cp_straight_yacht[:, 2]
    z_as_girths = (sail_cp_straight_yacht_z - tack_mounting[2]) / (max(sail_cp_straight_yacht_z) - tack_mounting[2])
    return z_as_girths

# circular import
#  csys_transformations: CSYS_transformations
# sail_set : SailSet problem with import
def get_cp_z_as_girths_all(sail_set : SailSet, csys_transformations : CSYS_transformations, center_of_pressure : np.ndarray) -> np.ndarray: 
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
def get_cp_z_as_girths_all_above(cp_z_as_girths_all : np.ndarray, sail_set : SailSet) -> Tuple[np.ndarray, np.ndarray]:
    """
    get_cp_z_as_girths_all_above get center of pressure z coordinate as girths above water

    :param np.ndarray cp_z_as_girths_all: center of pressure z coordinate as girths
    :param SailSet sail_set: sail set 
    :return Tuple[np.ndarray, np.ndarray]:  center of pressure z coordinate as girths above water for all panels, names of all panels
    """
    # 2 bo mamy odbicie 
    # parzyste numery to sa te nad woda
    n = 2 * len(sail_set.sails)
    z_splitted = np.split(cp_z_as_girths_all, n)
    # [::2] gets every second element from array z_splitted
    cp_z_as_girths_all_above = np.asarray(z_splitted[::2]).flatten()
    # cp_z_as_girths_all_above size
    repeat = int(cp_z_as_girths_all_above.shape[0] / len(sail_set.sails))
    names = np.array([])
    
    for sail in sail_set.sails:
        rep_names = np.repeat(sail.name, repeat)
        names = np.append(names, rep_names)
      
    return cp_z_as_girths_all_above, names      


def plot_mesh(mesh1 : np.array, mesh2 : np.array = None, show : bool = False, dimentions : List  = [0, 1, 2], color1 : str = 'green', color2 : str = None, title : str = '3d plot'):

    f = plt.figure(figsize=(12, 12))
    labels = {0: 'X', 1: 'Y', 2: 'Z'}
    ax = plt.axes(projection='3d')
    if len(dimentions) == 2:
        ax = plt.axes()
    
    ax.set_title(title)
    
    if len(dimentions) == 2:
        ax.set_xlabel(labels[dimentions[0]])
        ax.set_ylabel(labels[dimentions[1]])
    elif len(dimentions) == 3:
        ax.set_xlabel(labels[dimentions[0]])
        ax.set_ylabel(labels[dimentions[1]])
        ax.set_zlabel(labels[dimentions[2]])


    for i in range(mesh1.shape[0]):
        if len(dimentions) == 2:
            ax.plot(mesh1[i, :, dimentions[0]], mesh1[i, :, dimentions[1]], color1)
            if mesh2 is not None:
                ax.plot(mesh2[i, :, dimentions[0]], mesh2[i, :, dimentions[1]], color2)
        else:
            ax.plot3D(mesh1[i, :, 0], mesh1[i, :, 1], mesh1[i, :, 2], color1)
            if mesh2 is not None:
                ax.plot3D(mesh2[i, :, 0], mesh2[i, :, 1], mesh2[i, :, 2], color2)

    if show:
        plt.show()