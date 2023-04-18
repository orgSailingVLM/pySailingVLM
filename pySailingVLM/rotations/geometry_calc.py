import numpy as np
import math

from scipy.spatial.transform import Rotation as R
from numpy.linalg import multi_dot, norm

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    
    example
    v = [3, 5, 0]
    axis = [4, 4, 1]
    theta = 1.2 
    
    np.dot(rotation_matrix(axis,theta), v)
    # [ 2.74911638  4.77180932  1.91629719]
    
    Ry = rotation_matrix([0,1,0], np.deg2rad(45))
    np.dot(Ry, [1,456,1]) 
    # [  1.41421356e+00   4.56000000e+02  -1.11022302e-16]

    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

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
  
    l = norm(axis)
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


def rotate_points_around_origin_axis(ps : np.ndarray, axis : np.array, theta : float) -> np.ndarray:
    """
    rotate_points_around_origin_axis rotate points around axis (from origin)

    :param np.ndarray ps: array with points
    :param np.array axis: rotation axis
    :param float theta: angle in radians
    :return np.ndarray: rotated points
    """
    # origin is inside np.array([0, 0, 0])
    axis = axis / norm(axis)
    r = R.from_rotvec(theta * axis)
    return r.apply(ps)


