import numpy as np

from sailing_vlm.solver import mesher
from sailing_vlm.rotations import geometry_calc

from typing import List, Tuple
import numba

# to be used

def make_panels_from_mesh_spanwise(mesh : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    make_panels_from_mesh_spanwise make panels from spanwise mesh

    :param np.ndarray mesh: mesh array
    :return Tuple[np.ndarray, np.ndarray, np.ndarray]: panels, trailing_edge_info, leading_edge_info
    """
    n_lines = mesh.shape[0]
    n_points_per_line = mesh.shape[1]
    M = n_points_per_line - 1
    N = n_lines - 1
    
    panels = np.zeros((N * M, 4, 3))
    trailing_edge_info = np.full(N * M, False, dtype=bool)
    leading_edge_info = np.full(N * M, False, dtype=bool)
    
    counter = 0
    for i in range(N):
        for j in range(M):
            pSE = mesh[i + 1][j]
            pSW = mesh[i][j]
            pNW = mesh[i][j + 1]
            pNE = mesh[i + 1][j + 1]
            panels[counter] = [pSE, pSW, pNW, pNE] 

            if i == (n_lines - 2):
                trailing_edge_info[counter] = True
                
            else:
                if i == 0:
                    leading_edge_info[counter] = True
                
            counter += 1
            
    return  panels, trailing_edge_info, leading_edge_info


def make_panels_from_le_te_points(points : np.ndarray, grid_size :List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    make_panels_from_le_te_points make_panels_from_le_te_points make panels from leading end trailing edges points


    :param np.ndarray points: points
    :param List[int, int] grid_size: grid size
    :return Tuple[np.ndarray, np.ndarray, np.ndarray]: panels, trailing edge info, leading edge info
    """

    le_SW, te_SE, le_NW, te_NE = points
    nc, ns = grid_size
    south_line = mesher.discrete_segment(le_SW, te_SE, nc)
    north_line = mesher.discrete_segment(le_NW, te_NE, nc)

    mesh = mesher.make_point_mesh(south_line, north_line, ns)
    panels, trailing_edge_info, leading_edge_info = make_panels_from_mesh_spanwise(mesh)
    return panels, trailing_edge_info, leading_edge_info


def make_panels_from_le_points_and_chords(le_points : List[np.ndarray], grid_size : List[int], chords_vec : np.ndarray, interpolated_camber, interpolated_distance_from_LE, gamma_orientation : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    make_panels_from_le_points_and_chords make panels from leading edges and chords

    :param List[np.ndarray] le_points: leading edges points
    :param List[int] grid_size: zise of grid
    :param np.ndarray chords_vec: chords vectors
    :param float gamma_orientation: gamma orientation
    :return Tuple[np.ndarray, np.ndarray, np.ndarray]: panels, trailing_edge_info, leading_edge_info
    """
    
    le_SW,  le_NW = le_points
    n_chordwise, n_spanwise = grid_size
    le_line = mesher.discrete_segment(le_SW, le_NW, n_spanwise)
    te_line = np.copy(le_line)  # deep copy
    te_line += chords_vec

    ###
    # moje dodatki
   
    #mesh = mesher.make_airfoil_mesh(le_line, te_line, n_chordwise + 1, interpolated_distance_from_LE, interpolated_camber)
    ###
    
    #mesh = mesher.make_point_mesh(le_line, te_line, n_chordwise)
    
    # do testowania 
    # import matplotlib.pyplot as plt  
    # ax = plt.axes(projection='3d')
    
    # m = mesh.shape[0] * mesh.shape[1]
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    
    # for i in range(mesh.shape[0]):
    #     ax.plot(mesh[i, :, 0], mesh[i, :, 1], mesh[i, :, 2])
    # #ax.plot(mesh.reshape(m,3)[:,0], mesh.reshape(m,3)[:,1], mesh.reshape(m,3)[:,2])
    # #ax.scatter3D(mesh.reshape(m,3)[:,0], mesh.reshape(m,3)[:,1], mesh.reshape(m,3)[:,2])
    
    
    mesh = np.swapaxes(mesh, 0, 1)
    panels, trailing_edge_info, leading_edge_info = make_panels_from_mesh_spanwise(mesh)
    
    return panels, trailing_edge_info, leading_edge_info


# no seppedup
#@numba.jit(nopython=True, parallel=True)
def get_panels_area(panels: np.ndarray)-> np.ndarray:
    """
    get_panels_area get panels area

    :param np.ndarray panels: panels
    :return np.ndarray: areas
    """

    m = panels.shape[0]
    areas = np.zeros(m)

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