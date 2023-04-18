import numpy as np
from pySailingVLM.yacht_geometry.sail_geometry import SailGeometry
from pySailingVLM.rotations.csys_transformations import CSYS_transformations


class SailFactory:
    def __init__(self, csys_transformations: CSYS_transformations, n_spanwise=10, n_chordwise=1, rake_deg=90, sheer_above_waterline=0):
        # sin(pi-alpha) = sin(alpha)
        # cos(pi-alpha) = -cos(alpha)

        self._n_spanwise = n_spanwise  # number of panels (span-wise, above the water) per one sail
        self._n_chordwise = n_chordwise  # number of panels (chord-wise, above the water) per one sail
        self.rake = np.deg2rad(rake_deg)
        self.csys_transformations = csys_transformations
        self.sheer_above_waterline_xyz = [-sheer_above_waterline * np.cos(self.rake),
                                          0.,
                                          sheer_above_waterline * np.sin(self.rake)]

    def make_main_sail(self, main_sail_luff, boom_above_sheer, main_sail_chords=None, sail_twist_deg=None, LLT_twist=None, interpolated_camber=None, interpolated_distance_from_luff=None):
        boom_above_sheer_xyz = [-boom_above_sheer * np.cos(self.rake),
                                0.,
                                boom_above_sheer * np.sin(self.rake)]

        # define sail mounting points
        tack_mounting = np.array([self.sheer_above_waterline_xyz[0]+boom_above_sheer_xyz[0],
                                  0.,
                                  self.sheer_above_waterline_xyz[2]+boom_above_sheer_xyz[2]])

        head_mounting = np.array([tack_mounting[0]-main_sail_luff*np.cos(self.rake),
                                  0.,
                                  tack_mounting[2]+main_sail_luff*np.sin(self.rake)])

        main_sail = SailGeometry(head_mounting, tack_mounting, self.csys_transformations,
                                 n_spanwise=self._n_spanwise, n_chordwise=self._n_chordwise,
                                 chords=main_sail_chords,
                                 initial_sail_twist_deg=sail_twist_deg, name="main_sail", LLT_twist=LLT_twist, interpolated_camber=interpolated_camber, interpolated_distance_from_luff=interpolated_distance_from_luff)
        return main_sail

    def make_jib(self, jib_luff, foretriangle_base, foretriangle_height, jib_chords=None, sail_twist_deg=None, mast_LOA=0, LLT_twist=None, interpolated_camber=None, interpolated_distance_from_luff=None):
        # cosine theorem
        forestay_length = np.sqrt(foretriangle_base * foretriangle_base
                                  + foretriangle_height * foretriangle_height
                                  - 2 * foretriangle_base * foretriangle_height * np.cos(self.rake))
        # sine theorem
        forestay_angle = np.arcsin(foretriangle_height * np.sin(self.rake) / forestay_length)

        # define sail mounting points
        tack_mounting = np.array([-foretriangle_base + self.sheer_above_waterline_xyz[0] - mast_LOA/np.sin(self.rake),
                                  0.,
                                  self.sheer_above_waterline_xyz[2]])

        head_mounting = np.array([tack_mounting[0] + jib_luff*np.cos(forestay_angle),
                                  0.,
                                  tack_mounting[2] + jib_luff*np.sin(forestay_angle)])

        jib = SailGeometry(head_mounting, tack_mounting,  self.csys_transformations,
                           n_spanwise=self._n_spanwise, n_chordwise=self._n_chordwise,
                           chords=jib_chords,
                           initial_sail_twist_deg=sail_twist_deg, name="jib", LLT_twist=LLT_twist, interpolated_camber=interpolated_camber, interpolated_distance_from_luff=interpolated_distance_from_luff)
        return jib
