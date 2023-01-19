import numpy as np
import math
from numpy.linalg import multi_dot


def rotate_points_around_arbitrary_axis(ps : np.ndarray, p1 : np.array, p2 : np.array, theta : float) -> np.ndarray:
    # see https://www.engr.uvic.ca/~mech410/lectures/4_2_RotateArbi.pdf
    """
    rotate_points_around_arbitrary_axis translates vectors (with points) to origin, do the rotation around axis and translate back to original positions

    :param np.ndarray ps: array with points
    :param np.array p1: first point describing axis
    :param np.array p2: second point describing axis
    :param float theta: angle in radians
    :return np.ndarray: rotated points
    """

    ps = np.vstack((ps.transpose(),np.array([1]*ps.transpose().shape[1])))

    x1, y1, z1 = p1
    axis = p2 - p1
  
    l = np.linalg.norm(axis)
    a,b,c = axis
    v = np.sqrt(b**2 + c**2)

    D = np.array([[1,0,0,-x1],[0,1,0,-y1],[0,0,1,-z1],[0,0,0,1]])
    D_inv = np.array([[1,0,0,x1],[0,1,0,y1],[0,0,1,z1],[0,0,0,1]])

    R_x = np.array([[1,0,0,0],[0,c/v,-b/v,0],[0,b/v,c/v,0],[0,0,0,1]])
    R_x_inv = np.array([[1,0,0,0],[0,c/v,b/v,0],[0,-b/v,c/v,0],[0,0,0,1]])

    R_y = np.array([[v/l,0,-a/l,0],[0,1,0,0],[a/l,0,v/l,0],[0,0,0,1]])
    R_y_inv = np.array([[v/l,0,a/l,0],[0,1,0,0],[-a/l,0,v/l,0],[0,0,0,1]])

    ct = math.cos(theta)
    st = math.sin(theta)
    R_z = np.array([[ct,-st,0,0],[st,ct,0,0],[0,0,1,0],[0,0,0,1]])

    ps_new = multi_dot([D_inv, R_x_inv, R_y_inv, R_z, R_y, R_x, D, ps])
    return ps_new[:3].transpose()

# dziala
# x1 = np.array([6, -2, 0])
# x2 = np.array([12, 8, 0])
# p = np.array([[3, 5, 0], [10, 6, 0]])
x1 = np.array([2, 0, 3/2])
x2 = np.array([1, 1, 1])
#[-1, 3, 0]
p = np.array([[3, -1, 2]])
p2 = rotate_points_around_arbitrary_axis(p, x1, x2, np.pi / 3)
print(p2)