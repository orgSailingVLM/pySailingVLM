import numpy as np

from scipy.spatial.transform import Rotation as R

from pySailingVLM.solver.additional_functions import extract_above_water_quantities
from pySailingVLM.rotations.geometry_calc import rotation_matrix

class CSYS_transformations:
    def __init__(self, heel_deg=0, leeway_deg=0, v_from_original_xyz_2_reference_csys_xyz=np.array([0, 0, 0])):
        """
        v_from_original_2_reference_csys: np.array([reference_level_for_yaw_moment, 0, reference_level_for_heeling_moment])
        """
        self.heeling_rotation_matrix = rotation_matrix([1, 0, 0], np.deg2rad(-heel_deg))  # first rotation
        self.reverse_heeling_rotation_matrix = rotation_matrix([1, 0, 0], np.deg2rad(heel_deg))  # first rotation - for the mirror part
        self.leeway_rotation_matrix = rotation_matrix([0, 0, 1], np.deg2rad(leeway_deg))  # second rotation
        self.reverse_leeway_rotation_matrix = rotation_matrix([0, 0, 1], np.deg2rad(-leeway_deg))
        
        axis = [1, 0, 0]
        axis = axis / np.linalg.norm(axis)
        r = R.from_rotvec(np.deg2rad(-heel_deg) * axis)
        np.testing.assert_almost_equal(self.heeling_rotation_matrix, r.as_matrix())
        
        axis = [1, 0, 0]
        axis = axis / np.linalg.norm(axis)
        r = R.from_rotvec(np.deg2rad(heel_deg) * axis)
        np.testing.assert_almost_equal(self.reverse_heeling_rotation_matrix, r.as_matrix())
        
        axis = [0, 0, 1]
        axis = axis / np.linalg.norm(axis)
        r = R.from_rotvec(np.deg2rad(leeway_deg) * axis)
        np.testing.assert_almost_equal(self.leeway_rotation_matrix, r.as_matrix())
        
        axis = [0, 0, 1]
        axis = axis / np.linalg.norm(axis)
        r = R.from_rotvec(np.deg2rad(-leeway_deg) * axis)
        np.testing.assert_almost_equal(self.reverse_leeway_rotation_matrix, r.as_matrix())
        
        self.__leeway_deg = leeway_deg
        self.__heel_deg = heel_deg

        self.v_from_original_xyz_2_reference_csys_xyz = v_from_original_xyz_2_reference_csys_xyz

    def rotate_point_with_mirror(self, point):
        """
        from upright csys to heeled csys
        """
        if point[2] > 0:
            point = np.dot(self.heeling_rotation_matrix, point)
        else:
            point = np.dot(self.reverse_heeling_rotation_matrix, point)

        point = np.dot(self.leeway_rotation_matrix, point)
        return point

    def reverse_rotations_with_mirror(self, point):
        """
        from heeled csys to upright csys
        """
        point = np.dot(self.reverse_leeway_rotation_matrix, point)
        if point[2] > 0:
            point = np.dot(self.reverse_heeling_rotation_matrix, point)
        else:
            point = np.dot(self.heeling_rotation_matrix, point)

        return point

    def rotate_point_without_mirror(self, point):
        """
        from upright csys to heeled csys
        """
        point = np.dot(self.heeling_rotation_matrix, point)
        point = np.dot(self.leeway_rotation_matrix, point)
        return point

    def from_xyz_to_centerline_csys(self, array_of_xyz_vectors):
        """
        the x, y axes are parallel to water level
        the x axis is along the center line of the yacht, positive towards the stern
        the z axis is perpendicular to water level, positive upwards
        """
        array_of_vectors_in_centerline_csys = np.array([np.dot(self.leeway_rotation_matrix, v_xyz) for v_xyz in array_of_xyz_vectors])
        return array_of_vectors_in_centerline_csys

    def calc_centerline_moments(self, Mxyz):
        # side moment is around x axis
        # under water (mirror) moments are positive (righting)
        # above water moments are negative (heeling)
        M_centerline_csys = self.from_xyz_to_centerline_csys(Mxyz)
        _, total_above_water_moments_in_centerline_csys = extract_above_water_quantities(M_centerline_csys)
        return M_centerline_csys, total_above_water_moments_in_centerline_csys

    @property
    def leeway_deg(self):
        return self.__leeway_deg

    @property
    def heel_deg(self):
        return self.__heel_deg
