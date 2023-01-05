import numpy as np
import matplotlib.pyplot as plt 

from abc import abstractmethod, ABC
from sailing_vlm.rotations.geometry_calc import rotation_matrix

from sailing_vlm.rotations.csys_transformations import CSYS_transformations

from typing import List

from sailing_vlm.solver.panels import make_panels_from_le_points_and_chords
from sailing_vlm.solver.mesher import make_airfoil_mesh
class BaseGeometry:

    @property
    @abstractmethod
    def panels(self):
        pass

    
class SailGeometry(BaseGeometry, ABC):
    def __init__(self, head_mounting: np.array, tack_mounting: np.array,
                 csys_transformations: CSYS_transformations,
                 n_spanwise=10, n_chordwise=1, chords=None,
                 initial_sail_twist_deg=None, name=None, LLT_twist=None,  interpolated_camber=None, interpolated_distance_from_LE=None
                 ):

        self.__n_spanwise = n_spanwise  # number of panels (span-wise) - above the water
        self.__n_chordwise = n_chordwise  # number of panels (chord-wise) - in LLT there is line instead of panels
        self.name = name

        self.csys_transformations = csys_transformations

        self.head_mounting = head_mounting
        self.tack_mounting = tack_mounting
        """
            The geometry is described using the following CSYS.
            le - leading edge (luff) of the sail
            te - trailing edge (leech) of the sail
            below is example for the main sail.
            same for jib.

                        Z ^ (mast)     
                          |
                         /|
                        / |
                       /  |              ^ Y     
                   lem_NW +--+tem_NE    / 
                     /    |   \        /
                    /     |    \      /
                   /      |     \    /
                  /       |      \  /
                 /        |       \/
                /         |       /\
               /          |      /  \
              /           |     /    \
             /            |    /      \
            /      lem_SW |---/--------+tem_SE
           /              |  /          
  (bow) ------------------|-/-------------------------| (stern)
         \                |/                          |
    ------\---------------*---------------------------|-------------------------> X (water level)

        """

        le_NW = head_mounting
        le_SW = tack_mounting

        # mirror z coord in water surface
        # remember that direction of the lifting-line matters
        le_NW_underwater = np.array(
            [tack_mounting[0], tack_mounting[1], -tack_mounting[2]])  # leading edge South - West coordinate - mirror
        le_SW_underwater = np.array(
            [head_mounting[0], head_mounting[1], -head_mounting[2]])  # leading edge North - West coordinate - mirror
        
        
        chords_vec = np.array([chords, np.zeros(len(chords)), np.zeros(len(chords))])
        chords_vec = chords_vec.transpose()

        #### camber line tests
        
        mesh = make_airfoil_mesh([le_SW, le_NW],[self.__n_chordwise, self.__n_spanwise],chords_vec, interpolated_distance_from_LE, interpolated_camber)
        ####
        f = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax = f1.add_axes(rect=[0.0, 0.0, 0.1, 0.1], projection='3d')
        m = mesh.shape[0] * mesh.shape[1]
    
        
        
        # for i in range(mesh.shape[0]):
        #     ax.plot(mesh[i, :, 0], mesh[i, :, 1], mesh[i, :, 2])
        mesh_test = mesh.reshape(mesh.shape[0]* mesh.shape[1], 3)
        ax.plot(mesh_test[:, 0], mesh_test[:, 1], mesh_test[:, 2], '*')
        ########## 2D PLOT #########
        sh = mesh.shape[0] * mesh.shape[1]
        f2 = plt.figure()
        
        ax2 = f2.add_subplot(1, 1, 1)
        ax2.set_xlim([0, 1])
        ax2.axis('equal')
        ax2.grid()

        plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.98, wspace=None, hspace=None)
        
        
        #ax2.plot(mesh[:, :, 0].reshape(sh), mesh[:, :, 2].reshape(sh))
        for i in range(mesh.shape[0]):
            ax2.plot(mesh[i, :, 0], mesh[i, :, 2])
        
        
        mesh = mesh.reshape(mesh.shape[0]* mesh.shape[1], 3)
        mesh_rotated = np.array([self.csys_transformations.rotate_point_with_mirror(x) for x in mesh])
        
        ########## 3D PLOT #########
        
        f3 = plt.figure(figsize=(12, 12))
        #ax3 = plt.axes(projection='3d')
        ax3 = f3.add_subplot(1, 1, 1, projection='3d')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        # mesh_rotated.shape[0]
        # for i in range(100):
        #     ax3.plot(mesh_rotated[i, 0], mesh_rotated[i, 1], mesh_rotated[i, 2])
        ax3.plot(mesh_rotated[:,0], mesh_rotated[:,1], mesh_rotated[:,2], '*')
        plt.show()
        
        
        le_NW = self.csys_transformations.rotate_point_with_mirror(le_NW)
        le_SW = self.csys_transformations.rotate_point_with_mirror(le_SW)
        le_SW_underwater = self.csys_transformations.rotate_point_with_mirror(le_SW_underwater)
        le_NW_underwater = self.csys_transformations.rotate_point_with_mirror(le_NW_underwater)
        
        rchords_vec = np.array([self.csys_transformations.rotate_point_with_mirror(c) for c in chords_vec])
        frchords_vec = np.flip(rchords_vec, axis=0)

        if initial_sail_twist_deg is not None and LLT_twist is not None:
            print(f"Applying initial_sail_twist_deg to {self.name} -  Lifting Line, mode: {LLT_twist}")
            twist_dict = {
                'sheeting_angle_const': np.full(len(initial_sail_twist_deg), np.average(initial_sail_twist_deg)),
                'average_const': np.full(len(initial_sail_twist_deg), np.average(initial_sail_twist_deg)),
                'real_twist': initial_sail_twist_deg
            }
            sail_twist_deg = twist_dict[LLT_twist]
            axis = le_NW - le_SW  # head - tack
            rchords_vec = self.rotate_chord_around_le(axis, rchords_vec, sail_twist_deg)
            underwater_axis = le_NW_underwater - le_SW_underwater  # head - tack
            frchords_vec = self.rotate_chord_around_le(underwater_axis, frchords_vec,
                                                    np.flip(sail_twist_deg, axis=0))
             
            

            new_approach_panels, trailing_edge_info, leading_edge_info = make_panels_from_le_points_and_chords(
                [le_SW, le_NW],
                [self.__n_chordwise, self.__n_spanwise],
                rchords_vec, interpolated_camber, interpolated_distance_from_LE, gamma_orientation=-1)

            new_approach_panels_mirror, trailing_edge_info, leading_edge_info = make_panels_from_le_points_and_chords(
                [le_SW_underwater, le_NW_underwater],
                [self.__n_chordwise, self.__n_spanwise],
                frchords_vec, interpolated_camber, interpolated_distance_from_LE, gamma_orientation=-1)

        self.__panels_above = new_approach_panels
        self.__panels_under = new_approach_panels_mirror
        self.__panels = np.concatenate([self.__panels_above, self.__panels_under])
        
        # both trailing_edge_info and leading_edge_info are the same for above and underwater
        self.__trailing_edge_info = trailing_edge_info
        self.__leading_edge_info = leading_edge_info

        
        
    def rotate_chord_around_le(self, axis, chords_vec, sail_twist_deg_vec):
        # sail_twist = np.deg2rad(45.)
        # todo: dont forget to reverse rotations in postprocessing (plots)

        # m = rotation_matrix(axis, np.deg2rad(sail_twist_deg))
        rchords_vec = np.array([
            np.dot(rotation_matrix(axis, np.deg2rad(t)), c) for t, c in zip(sail_twist_deg_vec, chords_vec)])

        return rchords_vec
    
    @property
    def n_chordwise(self):
        return self.__n_chordwise
    
    @property
    def n_spanwise(self):
        return self.__n_spanwise
    
    @property
    def panels(self):
        return self.__panels
    
    @property
    def panels_above(self):
        return self.__panels_above
    
    @property
    def panels_under(self):
        return self.__panels_under
    
    @property
    def trailing_edge_info(self):
        return self.__trailing_edge_info
    
    @property
    def leading_edge_info(self):
        return self.__leading_edge_info
    

class SailSet(BaseGeometry):
    def __init__(self, sails: List[SailGeometry]):
        self.sails = sails
    
        panels_above = np.concatenate([sail.panels_above for sail in self.sails])
        panels_under = np.concatenate([sail.panels_under for sail in self.sails])
        self.__panels = np.concatenate([panels_above, panels_under])
        
        trailing_info_above = np.concatenate([sail.trailing_edge_info for sail in self.sails])
        trailing_info_under = np.concatenate([sail.trailing_edge_info for sail in self.sails])
        self.__trailing_edge_info = np.concatenate([trailing_info_above, trailing_info_under])
        
        leading_info_above = np.concatenate([sail.leading_edge_info for sail in self.sails])
        leading_info_under = np.concatenate([sail.leading_edge_info for sail in self.sails])
        self.__leading_edge_info = np.concatenate([leading_info_above, leading_info_under])

    @property
    def panels(self):
        return self.__panels
    
    @property
    def trailing_edge_info(self):
        return self.__trailing_edge_info
    
    @property
    def leading_edge_info(self):
        return self.__leading_edge_info