import numpy as np
import matplotlib.pyplot as plt 

from abc import abstractmethod, ABC
from sailing_vlm.rotations.geometry_calc import rotation_matrix

from sailing_vlm.rotations.csys_transformations import CSYS_transformations

from typing import List

from sailing_vlm.solver.panels import make_panels_from_le_points_and_chords
from sailing_vlm.solver.mesher import make_airfoil_mesh
from sailing_vlm.solver.additional_functions import plot_mesh, plot_both
class BaseGeometry:

    @property
    @abstractmethod
    def panels(self):
        pass

    
class SailGeometry(BaseGeometry, ABC):
    def __init__(self, head_mounting: np.array, tack_mounting: np.array,
                 csys_transformations: CSYS_transformations,
                 n_spanwise=10, n_chordwise=1, chords=None,
                 initial_sail_twist_deg=None, initial_sail_twist_deg_new = None, name=None, LLT_twist=None,  interpolated_camber=None, interpolated_distance_from_LE=None
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
        #mesh_underwater = make_airfoil_mesh([le_SW_underwater, le_NW_underwater],[self.__n_chordwise, self.__n_spanwise],chords_vec, interpolated_distance_from_LE, interpolated_camber)
        
        # to potem zniknie, bedzie shape chordwise na spanwise na 3
        sh0, sh1, sh2 = mesh.shape
        # plot_mesh(mesh, False, title='3d before rotation', color='green')
        # plot_mesh(mesh, False, dimentions=[0,2], color='darkorange', title='2d before rotation')
        # plot_mesh(mesh, False, dimentions=[0,1], color='darkgoldenrod', title='2d before rotation')
       
        rmesh = np.array([self.csys_transformations.rotate_point_with_mirror(x) for panel in mesh for x in panel])
        rmesh_sh = rmesh.reshape(sh0, sh1, sh2)
        # plot_mesh(mesh_rotated, False, title='3d rotated', color='blue')
        # plot_mesh(mesh_rotated, False, dimentions=[0,2], color='navy', title='2d rotated')
        # plot_mesh(mesh_rotated, False, dimentions=[0,1], color='purple', title='2d rotated')
        
        
        ### end of plots 
        
        
        le_NW = self.csys_transformations.rotate_point_with_mirror(le_NW)
        le_SW = self.csys_transformations.rotate_point_with_mirror(le_SW)
        le_SW_underwater = self.csys_transformations.rotate_point_with_mirror(le_SW_underwater)
        le_NW_underwater = self.csys_transformations.rotate_point_with_mirror(le_NW_underwater)
        
        rchords_vec = np.array([self.csys_transformations.rotate_point_with_mirror(c) for c in chords_vec])
        frchords_vec = np.flip(rchords_vec, axis=0)
        
        frmesh = np.flip(rmesh, axis=0)
        frmesh_sh = frmesh.reshape(sh0, sh1, sh2)
        plot_both(rmesh,frmesh,  True, dimentions = [0, 1, 2], color1='green', color2='blue',title='Comparoson')
            
        ### moje dodatki
        # rmesh dziala chyba jak rchords_vect ???
        frmesh = np.flip(rmesh, axis=0)
        ### end of moje dodatki
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
            ## moje dodatki
            # sail_twist_deg potem wraca do tego co bylo
            # mnozymy razy 100 bo na sztywno generuje geste luki
            #sail_twist_deg_test = sail_twist_deg * 100
            sail_twist_deg_new = initial_sail_twist_deg_new#twist_dict[LLT_twist]
            rmesh = self.rotate_chord_around_le(axis, rmesh, sail_twist_deg_new)
            frmesh = self.rotate_chord_around_le(underwater_axis, frmesh,
                                                    np.flip(sail_twist_deg_new, axis=0))
            ## end of moje dodatki
            
            rmesh_resh = rmesh.reshape(sh0, sh1, sh2)
            frmesh_resh= frmesh.reshape(sh0, sh1, sh2)
            # plot_mesh(rmesh_resh, False, title='3d above water', color='deepskyblue')
            # plot_mesh(rmesh_resh, False, dimentions=[0,2], color='hotpink', title='2d above water')
            # plot_mesh(rmesh_resh, False, dimentions=[0,1], color='teal', title='2d above water')
        
            # plot_mesh(frmesh_resh, False, title='3d underwater', color='lightcoral')
            # plot_mesh(frmesh_resh, False, dimentions=[0,2], color='maroon', title='2d underwater')
            # plot_mesh(frmesh_resh, False, dimentions=[0,1], color='sienna', title='2d underwater')
        
            
            # plot_both(rmesh_resh,frmesh_resh,  True, dimentions = [0, 1, 2], color1='green', color2='blue',title='Comparoson')
            
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