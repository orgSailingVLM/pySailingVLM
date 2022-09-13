import numpy as np
import pandas as pd

from abc import abstractmethod
from sailingVLM.Rotations.geometry_calc import rotation_matrix
from sailingVLM.Solver import Panel
from sailingVLM.Rotations.CSYS_transformations import CSYS_transformations
from sailingVLM.Solver.TrailingEdgePanel import TrailingEdgePanel
from sailingVLM.Solver.mesher import make_panels_from_le_te_points, make_panels_from_le_points_and_chords
from typing import List


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
    def extract_data_above_water(self, data):
        pass

    @abstractmethod
    def extract_data_above_water_to_df(self, data):
        data_above_water = self.extract_data_above_water(data)
        return pd.DataFrame(data=data_above_water)

    def get_cp_points(self):
        return np.array([p.get_cp_position() for p in self.panels1d])

    @abstractmethod
    def sail_cp_to_girths(self):
        pass

    @abstractmethod
    def get_cp_points_upright(self):
        pass


class  SailGeometry(BaseGeometry):
    def __init__(self, head_mounting: np.array, tack_mounting: np.array,
                 csys_transformations: CSYS_transformations,
                 n_spanwise=10, n_chordwise=1, chords=None,
                 initial_sail_twist_deg=None, name=None, LLT_twist = None
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
                frchords_vec = self.rotate_chord_around_le(underwater_axis, frchords_vec, np.flip(sail_twist_deg, axis=0))
                pass
            # inicjalizacja zmiennych? sa w ifie przeciez

            panels, mesh, new_approach_panels, trailing_edge_info = make_panels_from_le_points_and_chords(
                [le_SW, le_NW],
                [self.__n_chordwise, self.__n_spanwise],
                rchords_vec, gamma_orientation=-1)

            panels_mirror, mesh_mirror, new_approach_panels_mirror, trailing_edge_info = make_panels_from_le_points_and_chords(
                [le_SW_underwater, le_NW_underwater],
                [self.__n_chordwise, self.__n_spanwise],
                frchords_vec, gamma_orientation=-1)
        else:
            # make a lifting line instead of panels
            te_NE = le_NW  # trailing edge North - East coordinate
            te_SE = le_SW  # trailing edge South - East coordinate

            panels, mesh, new_approach_panels, trailing_edge_info = make_panels_from_le_te_points(
                [le_SW, te_SE, le_NW, te_NE],
                [self.__n_chordwise, self.__n_spanwise], gamma_orientation=-1)

            te_NE_underwater = le_NW_underwater  # trailing edge North - East coordinate
            te_SE_underwater = le_SW_underwater  # trailing edge South - East coordinate

            panels_mirror, mesh_mirror, new_approach_panels_mirror = make_panels_from_le_te_points(
                [le_SW_underwater, te_SE_underwater, le_NW_underwater, te_NE_underwater],
                [self.__n_chordwise, self.__n_spanwise], gamma_orientation=-1)

        # https://stackoverflow.com/questions/33356442/when-should-i-use-hstack-vstack-vs-append-vs-concatenate-vs-column-stack
        self.__panels = np.hstack((panels_mirror, panels))
        
        
        self.__panels1D = self.__panels.flatten()
        self.__spans = np.array([panel.get_panel_span_at_cp() for panel in self.panels1d])
        
        # moje dodatki w ramach eksperymentow
        # panale nad woda, panele pod woda (2*M*N, 4, 3)
        #self.my_panels = np.concatenate([new_approach_panels, new_approach_panels_mirror])
        
        self.panels_above = new_approach_panels
        self.panels_under = new_approach_panels_mirror
        self.trailing_edge_info = trailing_edge_info
       
    

        print()
        
        
    def rotate_chord_around_le(self, axis, chords_vec, sail_twist_deg_vec):
        # sail_twist = np.deg2rad(45.)
        # todo: dont forget to reverse rotations in postprocessing (plots)

        # m = rotation_matrix(axis, np.deg2rad(sail_twist_deg))
        rchords_vec = np.array([
            np.dot(rotation_matrix(axis, np.deg2rad(t)), c) for t, c in zip(sail_twist_deg_vec, chords_vec)])

        return rchords_vec

    def get_cp_points_upright(self):
        cp_points = self.get_cp_points()
        cp_straight_yacht = np.array([self.csys_transformations.reverse_rotations_with_mirror(p) for p in cp_points])
        return cp_straight_yacht

    def sail_cp_to_girths(self):
        sail_cp_straight_yacht = self.get_cp_points_upright()
        tack_mounting = self.tack_mounting
        y = sail_cp_straight_yacht[:, 2]
        y_as_girths = (y - tack_mounting[2]) / (max(y) - tack_mounting[2])
        return y_as_girths

    @property
    def spans(self):
        return self.__spans

    @property
    def panels1d(self) -> np.array([Panel]):
        return self.__panels1D

    @property
    def panels(self) -> np.array([Panel]):
        return self.__panels

    def extract_data_above_water(self, data):
        N = len(self.panels1d)
        underwater_part = int(N/2)
        data_above_water = data[underwater_part:N]
        return data_above_water


class SailSet(BaseGeometry):
    def __init__(self, sails: List[SailGeometry]):
        self.sails = sails
        # https://stackoverflow.com/questions/33356442/when-should-i-use-hstack-vstack-vs-append-vs-concatenate-vs-column-stack
        self.__panels = np.hstack([sail.panels for sail in self.sails])
        self.__panels1D = self.__panels.flatten()
        self.__spans = np.array([panel.get_panel_span_at_cp() for panel in self.panels1d])

        panels_above = np.concatenate([sail.panels_above for sail in self.sails])
        panels_under = np.concatenate([sail.panels_under for sail in self.sails])
        self.my_panels = np.concatenate([panels_above, panels_under])
        
        trailing_info_above = np.concatenate([sail.trailing_edge_info for sail in self.sails])
        trailing_info_under = np.concatenate([sail.trailing_edge_info for sail in self.sails])
        self.trailing_edge_info = np.concatenate([trailing_info_above, trailing_info_under])
        
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

    def extract_data_by_id(self, data, sail_no):
        sail = self.sails[sail_no]
        n_sail = int(len(sail.panels1d))
        n_start_of_sail = sum([len(self.sails[i].panels1d) for i in range(sail_no)])
        sail_data_above_water = data[n_start_of_sail:n_start_of_sail+n_sail]
        return sail_data_above_water

    def extract_data_above_water_by_id(self, data, sail_no):
        sail = self.sails[sail_no]
        n_sail = len(sail.panels1d)
        n_start_of_sail = sum([len(self.sails[i].panels1d) for i in range(sail_no)])
        underwater_part_of_sail = int(n_sail / 2)
        sail_data_above_water = data[n_start_of_sail+underwater_part_of_sail:n_start_of_sail+n_sail]
        return sail_data_above_water

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
