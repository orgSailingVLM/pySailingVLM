import numpy as np
import matplotlib.pyplot as plt 

from abc import abstractmethod, ABC
from pySailingVLM.rotations.geometry_calc import rotate_points_around_arbitrary_axis

from pySailingVLM.rotations.csys_transformations import CSYS_transformations

from typing import List

from pySailingVLM.solver.panels import make_panels_from_mesh_spanwise
from pySailingVLM.solver.mesher import make_airfoil_mesh
#from pySailingVLM.solver.additional_functions import plot_mesh
class BaseGeometry:

    @property
    @abstractmethod
    def panels(self):
        pass

    
class SailGeometry(BaseGeometry, ABC):
    def __init__(self, head_mounting: np.ndarray, tack_mounting: np.ndarray,
                 csys_transformations: CSYS_transformations,
                 n_spanwise=10, n_chordwise=1, chords=None,
                 initial_sail_twist_deg=None, name=None, LLT_twist=None,  interpolated_camber=None, interpolated_distance_from_luff=None
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
        fchords_vec = np.flip(chords_vec, axis=0)
        
        #### state "zero"
        mesh = make_airfoil_mesh([le_SW, le_NW],[self.__n_chordwise, self.__n_spanwise],chords_vec, interpolated_distance_from_luff, interpolated_camber)
        ###
        zero_mesh = np.swapaxes(mesh, 0, 1)
        panels_above_zero, _, _= make_panels_from_mesh_spanwise(zero_mesh)
        ###
        
        sh0, sh1, sh2 = mesh.shape
        mesh = mesh.reshape(sh0*sh1, sh2)
        
        mesh_underwater = make_airfoil_mesh([le_SW_underwater, le_NW_underwater],[self.__n_chordwise, self.__n_spanwise],fchords_vec, interpolated_distance_from_luff, np.flip(interpolated_camber))
        ### zero mesh
        zero_mesh_under = np.swapaxes(mesh_underwater, 0, 1)
        panels_under_zero, _, _= make_panels_from_mesh_spanwise(zero_mesh_under)
        ### 
        mesh_underwater = mesh_underwater.reshape(sh0*sh1, sh2)
    
        self.panels_under_zero = panels_under_zero
        self. panels_above_zero = panels_above_zero
        ### end of zero mesh
        
        # rotation # for heel
        #mesh:  le NW p1 p2 ... te
        #       le p1 p2 ... te
        #       le p1 p2 ... te
        #       le SW p2 ... te
        rmesh = np.array([self.csys_transformations.rotate_point_with_mirror(point) for point in mesh])
        rmesh_underwater = np.array([self.csys_transformations.rotate_point_with_mirror(point) for point in mesh_underwater])

        mesh = rmesh
        mesh_underwater = rmesh_underwater
        
        ## twist
        if initial_sail_twist_deg is not None and LLT_twist is not None:
            le_NW = self.csys_transformations.rotate_point_with_mirror(le_NW)
            le_SW = self.csys_transformations.rotate_point_with_mirror(le_SW)
            le_SW_underwater = self.csys_transformations.rotate_point_with_mirror(le_SW_underwater)
            le_NW_underwater = self.csys_transformations.rotate_point_with_mirror(le_NW_underwater)
        
            # print(f"Applying initial_sail_twist_deg to {self.name} -  Lifting Line, mode: {LLT_twist}")
            twist_dict = {
                'sheeting_angle_const': np.full(len(initial_sail_twist_deg), np.min(initial_sail_twist_deg)),
                'average_const': np.full(len(initial_sail_twist_deg), np.average(initial_sail_twist_deg)),
                'real_twist': initial_sail_twist_deg
            }
            sail_twist_deg = twist_dict[LLT_twist]
            sail_twist_deg = np.hstack([initial_sail_twist_deg] * (sh1))
            sail_twist_deg = sail_twist_deg.reshape(sh1, sh0).transpose().flatten()

            p2 = mesh[::sh1][-1]
            p1 = mesh[::sh1][0]
            trmesh = self.rotate_points_around_le(mesh, p1, p2, sail_twist_deg)
            # check if points on forestay are not rotated (they are on axis of rotation)
            np.testing.assert_almost_equal(trmesh[::sh1], mesh[::sh1])
           
            p2_u = mesh_underwater[::sh1][-1]
            p1_u = mesh_underwater[::sh1][0]
            trmesh_underwater = self.rotate_points_around_le(mesh_underwater, p1_u, p2_u, np.flip(sail_twist_deg))
            # check if points on forestay are not rotated (they are on axis of rotation)
            np.testing.assert_almost_equal(trmesh_underwater[::sh1], mesh_underwater[::sh1])

            mesh = trmesh
            mesh_underwater = trmesh_underwater
     
        # come back to original shape 
        mesh = mesh.reshape(sh0, sh1, sh2)
        mesh = np.swapaxes(mesh, 0, 1)
        mesh_underwater = mesh_underwater.reshape(sh0, sh1, sh2)
        mesh_underwater = np.swapaxes(mesh_underwater, 0, 1)
        
        new_approach_panels, trailing_edge_info, leading_edge_info= make_panels_from_mesh_spanwise(mesh)
        new_approach_panels_mirror, trailing_edge_info_mirror, leading_edge_info_mirror = make_panels_from_mesh_spanwise(mesh_underwater)
        
        np.testing.assert_array_equal(trailing_edge_info, trailing_edge_info_mirror)
        np.testing.assert_array_equal(leading_edge_info, leading_edge_info_mirror)
        
        self.__panels_above = new_approach_panels
        self.__panels_under = new_approach_panels_mirror
        self.__panels = np.concatenate([self.__panels_above, self.__panels_under])
        
        # both trailing_edge_info and leading_edge_info are the same for above and underwater
        self.__trailing_edge_info = trailing_edge_info
        self.__leading_edge_info = leading_edge_info

        
        
    def rotate_points_around_le(self, points, p1, p2, sail_twist_deg_vec):
     
        # if all elements are the same -> only once do the calculation -> less matrix multiplications
        result = np.all(sail_twist_deg_vec == sail_twist_deg_vec[0])
        rotated_points = points
        if result:
            rotated_points = rotate_points_around_arbitrary_axis(points, p1, p2, np.deg2rad(sail_twist_deg_vec[0]))
        else:
            # squezee removes "1" diemntion
            # rotate_points_around_arbitrary_axis needs [[a, b, c]] or [[a, b, c], [e, f, g], ...] 
            rotated_points = np.array([np.squeeze(rotate_points_around_arbitrary_axis(np.array([point]), p1, p2, np.deg2rad(deg))) for point, deg in zip(points, sail_twist_deg_vec) ])

        return rotated_points
    
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

        # mesh without any rotation (in 2d, one cooridinate is 0)
        panels_above_zero = np.concatenate([sail.panels_above_zero for sail in self.sails])
        panels_under_zero = np.concatenate([sail.panels_under_zero for sail in self.sails])
        self.zero_mesh = np.concatenate([panels_above_zero, panels_under_zero])
    @property
    def panels(self):
        return self.__panels
    
    @property
    def trailing_edge_info(self):
        return self.__trailing_edge_info
    
    @property
    def leading_edge_info(self):
        return self.__leading_edge_info