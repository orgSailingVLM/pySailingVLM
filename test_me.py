import numpy as np
from sailing_vlm.rotations.geometry_calc import rotation_matrix_v2


from scipy.spatial.transform import Rotation
from numpy.linalg import norm


import numpy as np

import math

v = 2 * np.array([1, -1, 0.5]) #np.array([-1, 3, 0])
axis = np.array([1, -1, 0.5])
n = norm(axis)
axis = axis / norm(axis)
theta = 60 # degrees


#axis = axis / norm(axis)  # normalize the rotation vector first
q0 = math.cos(np.deg2rad(theta) * 0.5)
q1, q2, q3 = axis*math.sin(np.deg2rad(theta)/2.0)


quat = [q1, q2, q3, q0]
quat = [q1, q2, q3, 0.0]
rot = Rotation.from_quat(quat) # degrees=False

# metoda 2
rot2 = Rotation.from_rotvec(np.deg2rad(theta) * axis)

print(quat)
print(rot2.as_quat())
new_v = rot.apply(v)  
print(v)    # results in [2.74911638 4.77180932 1.91629719]
print(new_v)
print(rot.as_euler('zxy'))

