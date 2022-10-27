import numpy as np
import pandas as pd

from abc import abstractmethod, ABC
from sailingVLM.Rotations.geometry_calc import rotation_matrix
from sailingVLM.Solver import Panel
from sailingVLM.Rotations.CSYS_transformations import CSYS_transformations
from sailingVLM.Solver.TrailingEdgePanel import TrailingEdgePanel
from sailingVLM.Solver.mesher import make_panels_from_le_te_points, make_panels_from_le_points_and_chords
from typing import List

from sailingVLM.Solver.forces import get_stuff_from_panels

# np.set_printoptions(precision=3, suppress=True)

class BaseGeometry:
    @property
    @abstractmethod
    def spans(self):
        pass

    @property
    @abstractmethod
    def panels1d(self):
        pass

    @property
    @abstractmethod
    def panels(self):
        pass

    @abstractmethod
    def extract_data_above_water_to_df(self, data):
        pass

    @abstractmethod
    def sail_cp_to_girths(self):
        pass

    @abstractmethod
    def get_cp_points_upright(self):
        pass

    def get_cp_points(self):
        return get_stuff_from_panels(self.panels, 'cp_position', (self.panels.shape[0], self.panels.shape[1], 1))

    def get_cp_points1d(self):
        return get_stuff_from_panels(self.panels1d, 'cp_position', (self.panels1d.shape[0], 3))


    @property
    def pressures(self):
        return get_stuff_from_panels(self.panels, 'pressure', (self.panels.shape[0], self.panels.shape[1], 1))

    @property
    def forces_xyz(self):
        return get_stuff_from_panels(self.panels, 'force_xyz', (self.panels.shape[0], self.panels.shape[1], 3))

    @property
    def V_app_fs_at_cp(self):
        return get_stuff_from_panels(self.panels, 'V_app_fs_at_cp', (self.panels.shape[0], self.panels.shape[1], 3))
        # return get_V_app_fs_at_cp_from_panels(self.panels)

    @property
    def V_induced_at_cp(self):
        return get_stuff_from_panels(self.panels, 'V_induced_at_cp', (self.panels.shape[0], self.panels.shape[1], 3))
        # return get_V_induced_at_cp_from_panels(self.panels)


class SailGeometry(BaseGeometry, ABC):
    def __init__(self, head_mounting: np.array, tack_mounting: np.array,
                 csys_transformations: CSYS_transformations,
                 n_spanwise=10, n_chordwise=1, chords=None,
                 initial_sail_twist_deg=None, name=None, LLT_twist=None
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

        le_NW = self.csys_transformations.rotate_point_with_mirror(le_NW)
        le_SW = self.csys_transformations.rotate_point_with_mirror(le_SW)
        le_SW_underwater = self.csys_transformations.rotate_point_with_mirror(le_SW_underwater)
        le_NW_underwater = self.csys_transformations.rotate_point_with_mirror(le_NW_underwater)

        if chords is not None:
            chords_vec = np.array([chords, np.zeros(len(chords)), np.zeros(len(chords))])
            chords_vec = chords_vec.transpose()

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
                pass
            # inicjalizacja zmiennych? sa w ifie przeciez

            panels, mesh, new_approach_panels, trailing_edge_info, leading_edge_info = make_panels_from_le_points_and_chords(
                [le_SW, le_NW],
                [self.__n_chordwise, self.__n_spanwise],
                rchords_vec, gamma_orientation=-1)

            panels_mirror, mesh_mirror, new_approach_panels_mirror, trailing_edge_info, leading_edge_info = make_panels_from_le_points_and_chords(
                [le_SW_underwater, le_NW_underwater],
                [self.__n_chordwise, self.__n_spanwise],
                frchords_vec, gamma_orientation=-1)
        else:
            # make a lifting line instead of panels
            te_NE = le_NW  # trailing edge North - East coordinate
            te_SE = le_SW  # trailing edge South - East coordinate

            panels, mesh, new_approach_panels, trailing_edge_info, leading_edge_info = make_panels_from_le_te_points(
                [le_SW, te_SE, le_NW, te_NE],
                [self.__n_chordwise, self.__n_spanwise], gamma_orientation=-1)

            te_NE_underwater = le_NW_underwater  # trailing edge North - East coordinate
            te_SE_underwater = le_SW_underwater  # trailing edge South - East coordinate

            panels_mirror, mesh_mirror, new_approach_panels_mirror, trailing_edge_info, leading_edge_info = make_panels_from_le_te_points(
                [le_SW_underwater, te_SE_underwater, le_NW_underwater, te_NE_underwater],
                [self.__n_chordwise, self.__n_spanwise], gamma_orientation=-1)

        # https://stackoverflow.com/questions/33356442/when-should-i-use-hstack-vstack-vs-append-vs-concatenate-vs-column-stack
        # self.__panels = np.vstack((panels, panels_mirror))
        self.__panels = np.hstack((panels_mirror, panels))  # original version
        self.__panels1D = self.__panels.flatten()
        self.__spans = np.array([panel.get_panel_span_at_cp() for panel in self.panels1d])
        
        # moje dodatki w ramach eksperymentow
        
        self.panels_above = new_approach_panels
        self.panels_under = new_approach_panels_mirror
        
        
        self.my_panels = np.concatenate([self.panels_above, self.panels_under])
        
        # both trailing_edge_info and leading_edge_info are the same for above and underwater
        self.trailing_edge_info = trailing_edge_info
        self.leading_edge_info = leading_edge_info
    

        self.tack_mounting_arr = np.array([self.tack_mounting for i in range(self.__n_chordwise * self.__n_spanwise)])
        print()
        
        
    def rotate_chord_around_le(self, axis, chords_vec, sail_twist_deg_vec):
        # sail_twist = np.deg2rad(45.)
        # todo: dont forget to reverse rotations in postprocessing (plots)

        # m = rotation_matrix(axis, np.deg2rad(sail_twist_deg))
        rchords_vec = np.array([
            np.dot(rotation_matrix(axis, np.deg2rad(t)), c) for t, c in zip(sail_twist_deg_vec, chords_vec)])

        return rchords_vec
# powrot zagla do ukladu przy pomoscie 
    def get_cp_points_upright(self):
        cp_points = self.get_cp_points1d()
        cp_straight_yacht = np.array([self.csys_transformations.reverse_rotations_with_mirror(p) for p in cp_points])
        return cp_straight_yacht

    def sail_cp_to_girths(self):
        sail_cp_straight_yacht = self.get_cp_points_upright()
        # rog helsowy 
        tack_mounting = self.tack_mounting
        y = sail_cp_straight_yacht[:, 2]
        # ciesnienie w procentach (girths) liczac od rogu
        y_as_girths = (y - tack_mounting[2]) / (max(y) - tack_mounting[2])
        return y_as_girths

    @property
    def spans(self):
        return self.__spans
    
    @property
    def n_chordwise(self):
        return self.__n_chordwise
    
    @property
    def n_spanwise(self):
        return self.__n_spanwise
    
    @property
    def panels1d(self) -> np.array([Panel]):
        return self.__panels1D

    @property
    def panels(self) -> np.array([Panel]):
        return self.__panels


class SailSet(BaseGeometry):
    def __init__(self, sails: List[SailGeometry]):
        self.sails = sails
        # https://stackoverflow.com/questions/33356442/when-should-i-use-hstack-vstack-vs-append-vs-concatenate-vs-column-stack
        # self.__panels = np.vstack([sail.panels for sail in self.sails])
        self.__panels = np.hstack([sail.panels for sail in self.sails]) # original version
        self.__panels1D = self.__panels.flatten()
        self.__spans = np.array([panel.get_panel_span_at_cp() for panel in self.panels1d])

        panels_above = np.concatenate([sail.panels_above for sail in self.sails])
        panels_under = np.concatenate([sail.panels_under for sail in self.sails])
        self.my_panels = np.concatenate([panels_above, panels_under])
        
        trailing_info_above = np.concatenate([sail.trailing_edge_info for sail in self.sails])
        trailing_info_under = np.concatenate([sail.trailing_edge_info for sail in self.sails])
        self.trailing_edge_info = np.concatenate([trailing_info_above, trailing_info_under])
        
        leading_info_above = np.concatenate([sail.leading_edge_info for sail in self.sails])
        leading_info_under = np.concatenate([sail.leading_edge_info for sail in self.sails])
        self.leading_edge_info = np.concatenate([leading_info_above, leading_info_under])

        # nie uzywane chyba
        tack_mounting_arr_above = np.concatenate([sail.tack_mounting_arr for sail in self.sails])
        tack_mounting_arr_under = np.concatenate([sail.tack_mounting_arr for sail in self.sails])
        self.tack_mounting_arr = np.concatenate([tack_mounting_arr_above, tack_mounting_arr_under])
        
        print()
    @property
    def panels1d(self):
        return self.__panels1D

    @property
    def panels(self) -> np.array([Panel]):
        return self.__panels

    @property
    def spans(self):
        return self.__spans

    def get_cp_points_upright(self):
        cp_points_straight = np.array([[], [], []]).transpose()
        for sail in self.sails:
            cp_points_straight = np.append(cp_points_straight, sail.get_cp_points_upright(), axis=0)
        return cp_points_straight

    def sail_cp_to_girths(self):
        y_as_girths = np.array([])
        for sail in self.sails:
            y_as_girths = np.append(y_as_girths, sail.sail_cp_to_girths())
        return y_as_girths

    def extract_data_above_water_by_id(self, data, sail_no):
        all_cp_points = self.get_cp_points1d()[:, 2]
        reference_cp_points = self.sails[sail_no].get_cp_points1d()[:, 2]
        above_water_ref_cp_points = reference_cp_points[reference_cp_points > 0].transpose()
        index_array = np.array([np.where(all_cp_points == point)[0][0] for point in above_water_ref_cp_points])

        if isinstance(data, pd.DataFrame):
            above_water_quantities = data.iloc[index_array]
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                above_water_quantities = data[index_array]
            elif len(data.shape) == 2:
                above_water_quantities = data[index_array, :]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return above_water_quantities

    def extract_data_above_water_to_df(self, data):
        above_water_dfs = [pd.DataFrame(self.extract_data_above_water_by_id(data, i)) for i in range(len(self.sails))]
        merged_df_above_water = pd.concat(above_water_dfs)
        return merged_df_above_water

    def get_sail_name_for_each_element(self):
        names = []
        for i in range(len(self.sails)):
            for j in range(len(self.sails[i].panels1d)):
                names.append(self.sails[i].name)

        return pd.DataFrame({'sail_name': names})
