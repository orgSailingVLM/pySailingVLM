import numpy as np
import math


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


