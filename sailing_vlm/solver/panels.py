import numpy as np

from sailing_vlm.solver import mesher
from sailing_vlm.rotations import geometry_calc

import numba
# to be used

def make_panels_from_mesh_spanwise(mesh) -> np.array:
    n_lines = mesh.shape[0]
    n_points_per_line = mesh.shape[1]
    M = n_points_per_line - 1
    N = n_lines - 1
    
    new_approach_panels = np.zeros((N * M, 4, 3))
    trailing_edge_info = np.full(N * M, False, dtype=bool)
    leading_edge_info = np.full(N * M, False, dtype=bool)
    
    counter = 0
    for i in range(N):
        for j in range(M):
            pSE = mesh[i + 1][j]
            pSW = mesh[i][j]
            pNW = mesh[i][j + 1]
            pNE = mesh[i + 1][j + 1]
            new_approach_panels[counter] = [pSE, pSW, pNW, pNE] 

            if i == (n_lines - 2):
                trailing_edge_info[counter] = True
                
            else:
                if i == 0:
                    leading_edge_info[counter] = True
                
            counter += 1
            
    return  new_approach_panels, trailing_edge_info, leading_edge_info


def make_panels_from_le_te_points(points, grid_size, gamma_orientation):
    """
    this is the main meshing method
    :param points: 
    :param grid_size: 
    :return: 
    """
    le_SW, te_SE, le_NW, te_NE = points
    nc, ns = grid_size
    south_line = mesher.discrete_segment(le_SW, te_SE, nc)
    north_line = mesher.discrete_segment(le_NW, te_NE, nc)

    mesh = mesher.make_point_mesh(south_line, north_line, ns)
    new_approach_panels, trailing_edge_info, leading_edge_info = make_panels_from_mesh_spanwise(mesh)
    return new_approach_panels


def make_panels_from_le_points_and_chords(le_points, grid_size, chords_vec, gamma_orientation):
    le_SW,  le_NW = le_points
    n_chordwise, n_spanwise = grid_size
    le_line = mesher.discrete_segment(le_SW, le_NW, n_spanwise)
    te_line = np.copy(le_line)  # deep copy
    te_line += chords_vec

    mesh = mesher.make_point_mesh(le_line, te_line, n_chordwise)
    # panels = make_panels_from_mesh_chordwise(mesh)
    mesh = np.swapaxes(mesh, 0, 1)
    new_approach_panels, trailing_edge_info, leading_edge_info = make_panels_from_mesh_spanwise(mesh)
    return new_approach_panels, trailing_edge_info, leading_edge_info


def create_panels(half_wing_span : float, chord : float, AoA_deg : float, M : int, N : int) -> np.ndarray:
    le_NW = np.array([0., half_wing_span, 0.])      # leading edge North - West coordinate
    le_SW = np.array([0., -half_wing_span, 0.])     # leading edge South - West coordinate

    te_NE = np.array([chord, half_wing_span, 0.])   # trailing edge North - East coordinate
    te_SE = np.array([chord, -half_wing_span, 0.])  # trailing edge South - East coordinate

    Ry = geometry_calc.rotation_matrix([0, 1, 0], np.deg2rad(AoA_deg))

    ### MESH DENSITY ###
    # sprawdzci czy nie jest odrotnie M i N
    ns = M    # number of panels (spanwise)
    nc = N   # number of panels (chordwise)


    new_approach_panels = make_panels_from_le_te_points(
        [np.dot(Ry, le_SW),
        np.dot(Ry, te_SE),
        np.dot(Ry, le_NW),
        np.dot(Ry, te_NE)],
        [nc, ns],
        gamma_orientation=1)
    return new_approach_panels

# no seppedup
#@numba.jit(nopython=True, parallel=True)
def get_panels_area(panels: np.ndarray)-> np.ndarray:
    
    #m = N * M
    m = panels.shape[0]
    areas = np.zeros(m)
    #areas = np.zeros(m, dtype=float)
    sh = panels.shape[0]
    # numba.prange
    # range works slightly quicker than numba.prange (without decorator of course)
    for i in range(sh):
        
        p = [panels[i, 0], panels[i, 1], panels[i, 2], panels[i, 3]]
        path = []
        for j in range(len(p) - 1):
            step = p[j + 1] - p[j]
            path.append(step)

        area = 0
        for k in range(len(path) - 1):
            s = np.cross(path[k], path[k + 1])
            s = np.linalg.norm(s)
            area += 0.5 * s
        areas[i] = area   

    return areas