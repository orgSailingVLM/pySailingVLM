import numpy as np
from scipy import interpolate


class Interpolator:
    def __init__(self, interpolation_type):
        self.interpolation_type = interpolation_type

        self.interpolation_switch = {
            "linear": lambda x_new, x, y: np.interp(x_new, x, y),
            "spline": lambda x_new, x, y: interpolate.splev(x_new, interpolate.splrep(x, y, s=0), der=0),
        }

    def interpolate_girths(self, x, y, n_points):
        x_new = np.linspace(0., 1., num=n_points, endpoint=True)
        y_new = self.interpolation_switch[self.interpolation_type](x_new, x, y)
        return y_new

    def interpolate_and_mirror_girths(self, girths, y_data_for_interpolation, n_points):
        chord_sections = self.interpolate_girths(girths, y_data_for_interpolation, n_points)
        chord_sections_mirror = np.flip(chord_sections)
        return np.concatenate([chord_sections_mirror, chord_sections])

    def interpolate_data_for_sections(self, data_main_sail, data_jib, ms_girths, jib_girths, n_span):
        ms_interpolated_sections = [self.interpolate_and_mirror_girths(ms_girths, d, n_span) for d in data_main_sail]
        jib_interpolated_sections = [self.interpolate_and_mirror_girths(jib_girths, d, n_span) for d in data_jib]

        sailset_interpolated_sections = [np.concatenate([j, ms]) for j, ms in
                                         zip(jib_interpolated_sections, ms_interpolated_sections)]
        return sailset_interpolated_sections
