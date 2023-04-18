import numpy as np

from typing import List
from pySailingVLM.thin_airfoil.vlm_airfoil import VlmAirfoil
def discrete_segment(p1, p2, n):
    segment = []
    step = (p2 - p1) / n

    for i in range(n):
        point = p1 + i * step
        segment.append(point)

    segment.append(p2)
    return np.array(segment)


def make_point_mesh(segment1, segment2, n):
    mesh = []
    # TODO: dodac camber w stylu NACA airfoil --> wzorek z wikipedii
    # TODO: upewnic sie ze normalna jest w dobra strone dla pochylonej lodki
    for p1, p2 in zip(segment1, segment2):
        s = discrete_segment(p1, p2, n)
        mesh.append(np.array(s))

    return np.array(mesh)


def make_airfoil_mesh(le_points : List[np.ndarray], grid_size : List[int], chords_vec : np.ndarray, interpolated_distance_from_LE, interpolated_camber) -> np.ndarray:
    """
    make_airfoil_mesh make airfoil mesh before any rotation/twist etc

    :param List[np.ndarray] le_points: list with points on leading edge like [le_SW,  le_NW]
    :param List[int] grid_size: size of grid like [n_chordwise, n_spanwise]
    :param np.ndarray chords_vec: vector with chords
    :param _type_ interpolated_distance_from_LE: interpolated distance from leading edge
    :param _type_ interpolated_camber: interpolated camber
    :return np.ndarray: array with generated mesh
    """
    le_SW,  le_NW = le_points
    n_chordwise, n_spanwise = grid_size
    le_line = discrete_segment(le_SW, le_NW, n_spanwise)
    te_line = np.copy(le_line)  # deep copy
    te_line += chords_vec
    
    n = n_chordwise + 1
    segment1 = le_line
    segment2 = te_line
    distance = interpolated_distance_from_LE
    camber = interpolated_camber
    
    # segment1 , segemnt2 
    # http://www.airfoiltools.com/airfoil/naca4digit?MNaca4DigitForm%5Bcamber%5D=9&MNaca4DigitForm%5Bposition%5D=50&MNaca4DigitForm%5Bthick%5D=1&MNaca4DigitForm%5BnumPoints%5D=100&MNaca4DigitForm%5BcosSpace%5D=0&MNaca4DigitForm%5BcosSpace%5D=1&MNaca4DigitForm%5BcloseTe%5D=0&yt0=Plot
    # p is the position of the maximum camber divided by 10. In the example P=4 so the maximum camber is at 0.4 or 40% of the chord.
    # m is the maximum camber divided by 100. In the example M=2 so the camber is 0.02 or 2% of the chord
    mesh = []
    counter = 0
    for p1, p2 in zip(segment1, segment2):
        p = distance[counter]
        m = camber[counter] 
        foil = VlmAirfoil(m, p, 0.0, n=n)
        
        xs = (p2[0] - p1[0]) * foil.xc + p1[0]
        ys = (p2[0] - p1[0]) * foil.yc 
        zs =[p1[2]] * n 
    
        counter += 1
        mesh.append(list(zip(xs,ys, zs)))

    return np.array(mesh)
   
    