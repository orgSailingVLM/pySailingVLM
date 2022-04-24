import numpy as np
import math


def rotate_around_point(axis, theta, local_csys_origin, point):
    m = rotation_matrix(axis, theta)
    p_in_local_csys = point - local_csys_origin
    length = np.linalg.norm(p_in_local_csys, axis=0)
    point_rotated_in_local_csys = np.dot(m, p_in_local_csys)
    length2 = np.linalg.norm(point_rotated_in_local_csys, axis=0)
    point_in_xyz_csys = point_rotated_in_local_csys + local_csys_origin

    return point_in_xyz_csys


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


