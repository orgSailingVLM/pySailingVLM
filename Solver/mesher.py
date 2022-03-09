import numpy as np
from Solver.Panel import Panel
from Solver.TrailingEdgePanel import TrailingEdgePanel


def join_panels(panels1, panels2):
    joined_panels = []
    for p1, p2 in zip(panels1, panels2):
        temp = np.append(p1, p2)
        joined_panels.append(temp)
    return np.array(joined_panels)


def flat_panels(panels):
    joined_panels = []
    for p in panels:
        joined_panels.append(p.flatten())
    return np.array(joined_panels).flatten()


def make_panels_from_le_points_and_chords(le_points, grid_size, chords_vec, gamma_orientation):
    le_SW,  le_NW = le_points
    n_chordwise, n_spanwise = grid_size
    le_line = discrete_segment(le_SW, le_NW, n_spanwise)
    te_line = np.copy(le_line)  # deep copy
    te_line += chords_vec

    mesh = make_point_mesh(le_line, te_line, n_chordwise)
    # panels = make_panels_from_mesh_chordwise(mesh)
    mesh = np.swapaxes(mesh, 0, 1)
    panels = make_panels_from_mesh_spanwise(mesh, gamma_orientation)
    return panels, mesh


def make_panels_from_le_te_points(points, grid_size, gamma_orientation):
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


def make_panels_from_mesh_chordwise(mesh):
    panels = []

    # n_lines = mesh.shape[0]
    # n_points_per_line = mesh.shape[1]
    n_lines = mesh.shape[1]
    n_points_per_line = mesh.shape[0]
    for i in range(n_lines - 1):
        panels.append([])
        for j in range(n_points_per_line - 1):
            # pSW = mesh[i][j]
            # pNW = mesh[i + 1][j]
            # pSE = mesh[i][j + 1]
            # pNE = mesh[i + 1][j + 1]
            pSE = mesh[i + 1][j]
            pSW = mesh[i][j]
            pNW = mesh[i][j + 1]
            pNE = mesh[i + 1][j + 1]
            panel = Panel(p1=pSE,
                          p2=pSW,
                          p3=pNW,
                          p4=pNE)
            panels[i].append(panel)

    return np.array(panels)

# tutaj to rozpisac
def make_panels_from_mesh_spanwise(mesh, gamma_orientation) -> np.array([Panel]):
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
            # if last panel -> make trailing panel
            if i == (n_lines - 2):
                panel = TrailingEdgePanel(
                              p1=pSE,
                              p2=pSW,
                              p3=pNW,
                              p4=pNE,
                              gamma_orientation=gamma_orientation)
            else:
                panel = Panel(p1=pSE,
                              p2=pSW,
                              p3=pNW,
                              p4=pNE,
                              gamma_orientation=gamma_orientation)
            panels[i].append(panel)

    return np.array(panels)
