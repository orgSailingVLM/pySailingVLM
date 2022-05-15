import numpy as np

def my_make_panels_from_le_te_points(points, grid_size, gamma_orientation):
    """
    this is the main meshing method
    :param points: 
    :param grid_size: 
    :return: 
    """
    le_SW, te_SE, le_NW, te_NE = points
    nc, ns = grid_size
    south_line = discrete_segment(le_SW, te_SE, nc)
    north_line = discrete_segment(le_NW, te_NE, nc)

    mesh = make_point_mesh(south_line, north_line, ns)
    panels = make_panels_from_mesh_spanwise(mesh, gamma_orientation)
    return panels, mesh


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
    for p1, p2 in zip(segment1, segment2):
        s = discrete_segment(p1, p2, n)
        mesh.append(np.array(s))

    return np.array(mesh)



# tutaj to rozpisac
def make_panels_from_mesh_spanwise(mesh, gamma_orientation):
    panels = []

    n_lines = mesh.shape[0]
    n_points_per_line = mesh.shape[1]

    for i in range(n_lines - 1):
        panels.append([])
        for j in range(n_points_per_line - 1):
            pSE = mesh[i + 1][j]
            pSW = mesh[i][j]
            pNW = mesh[i][j + 1]
            pNE = mesh[i + 1][j + 1]
            
            panel = [pSE, pSW, pNW, pNE]
            panels[i].append(panel)
            
    return np.array(panels)
