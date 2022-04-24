import numpy as np
from airfoils import Airfoil
from sailingVLM.Rotations.CSYS_transformations import CSYS_transformations


class HullGeometry:
    """
    this class is to
    preview where the hull is in 3d plot
    estimate underwater_centreline_center_of_effort
    """
    def __init__(self, sheer_above_waterline, foretriangle_base,
                 csys_transformations: CSYS_transformations,
                 center_of_lateral_resistance_upright=np.array([0, 0, 0])):
        self.heel_leeway = csys_transformations

        foil = Airfoil.NACA4('1028')
        # foil.plot()
        n = 51
        chord_x = np.linspace(0, 1., num=n, endpoint=True)
        sheers_above_waterline = np.full(n, sheer_above_waterline)

        scale = foretriangle_base*3.
        deck_centerline = np.array([-1.1*foretriangle_base + scale*chord_x, np.zeros(n), sheers_above_waterline]).transpose()
        deck_port_line = np.array([-1.1*foretriangle_base + scale*chord_x, scale*np.flip(foil.y_upper(chord_x)), sheers_above_waterline]).transpose()  # left
        deck_starboard_line = np.array([-1.1*foretriangle_base + scale*chord_x, scale*np.flip(foil.y_lower(chord_x)), sheers_above_waterline]).transpose()   # right
        deck_port_line_underwater = np.array([-1.1*foretriangle_base + scale*chord_x, scale*np.flip(foil.y_upper(chord_x)), -1 * sheers_above_waterline]).transpose()  # left
        deck_starboard_line_underwater = np.array([-1.1*foretriangle_base + scale*chord_x, scale*np.flip(foil.y_lower(chord_x)), -1 * sheers_above_waterline]).transpose()   # right

        # apply rotations
        self.deck_centerline = np.array([csys_transformations.rotate_point_with_mirror(p) for p in deck_centerline])
        self.deck_port_line = np.array([csys_transformations.rotate_point_with_mirror(p) for p in deck_port_line])
        self.deck_starboard_line = np.array([csys_transformations.rotate_point_with_mirror(p) for p in deck_starboard_line])
        self.deck_port_line_underwater = np.array([csys_transformations.rotate_point_with_mirror(p) for p in deck_port_line_underwater])
        self.deck_starboard_line_underwater = np.array([csys_transformations.rotate_point_with_mirror(p) for p in deck_starboard_line_underwater])

        self.center_of_lateral_resistance = \
            csys_transformations.rotate_point_without_mirror(center_of_lateral_resistance_upright)  # remember to rotate it not as a mirror!
