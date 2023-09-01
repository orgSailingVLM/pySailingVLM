
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pySailingVLM.solver.vlm import Vlm
    from pySailingVLM.results.inviscid_flow import InviscidFlowResults
    from pySailingVLM.inlet.inlet_conditions import InletConditions
    from pySailingVLM.yacht_geometry.hull_geometry import HullGeometry
import numpy as np
import math
import sys
import scienceplots
import matplotlib.pyplot as plt

from pySailingVLM.solver.additional_functions import normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d as a3
import matplotlib.cm as cm
import matplotlib as mpl

from typing import Tuple

import spatialpandas as sp
import holoviews as hv
from holoviews.streams import PlotSize

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def display_panels_xyz(vlm : Vlm, color_panels_by : str) -> Tuple[plt.axes, int]:
    """
    display_panels_xyz display panels in 3 dimentions

    :param Vlm vlm: vlm class instance
    :param color_panels_by: gamma_magnitude or p_coeffs string
    :return List[plt.axes, int]: list containing matplotlib plt ax and size of water (int)
    """
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def set_ax_eq(ax : plt.Axes, X : np.ndarray, Y : np.ndarray, Z : np.ndarray) -> float:
        """
        set_ax_eq sets limits on x, y, z axes

        :param plt.Axes ax: ax to be modified
        :param np.ndarray X: x array
        :param np.ndarray Y: y array
        :param np.ndarray Z: z array
        :return float: max range number 
        """
        # ax.set_aspect('equal') - matplotlib bug
        # dirty hack: NotImplementedError: It is not currently possible to manually set the aspect on 3D axes
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return max_range

    data = None
    if color_panels_by == 'gamma_magnitude':
        data = vlm.pressure
    elif color_panels_by == 'p_coeffs':
        data = vlm.p_coeffs
    else:
        data = vlm.pressure
    norm = mpl.colors.Normalize(vmin=min(data), vmax=max(data))
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for vtx, d in zip(vlm.panels, data):
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(m.to_rgba(d))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

    fig.colorbar(m, ax=ax)
    ###
    # plot water level
    water_size = int(1.1*math.ceil(max(max(abs(vlm.leading_mid_points[:, 2])), max(abs(vlm.trailing_mid_points[:, 2])))))
    xx, yy = np.meshgrid(range(-water_size, water_size), range(-water_size, water_size))
    zz = 0 * xx + 0 * yy
    ax.plot_surface(xx, yy, zz, alpha=0.25)

    set_ax_eq(ax, xx, yy, zz)
    return ax, water_size

def display_panels_or_rings(things : np.ndarray, pressure : np.ndarray, leading_mid_points : np.ndarray, trailing_mid_points : np.ndarray) -> Tuple[plt.axes, int]:
    """
    display_panels_or_rings display panels or rings  in 3 dimentions

    :param Vlm vlm: vlm class instance
    :return List[plt.axes, int]: list containing matplotlib plt ax and size of water (int)
    """
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def set_ax_eq(ax : plt.Axes, X : np.ndarray, Y : np.ndarray, Z : np.ndarray) -> float:
        """
        set_ax_eq sets limits on x, y, z axes

        :param plt.Axes ax: ax to be modified
        :param np.ndarray X: x array
        :param np.ndarray Y: y array
        :param np.ndarray Z: z array
        :return float: max range number 
        """
        # ax.set_aspect('equal') - matplotlib bug
        # dirty hack: NotImplementedError: It is not currently possible to manually set the aspect on 3D axes
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return max_range


    #norm = mpl.colors.Normalize(vmin=min(pressure), vmax=max(pressure))
    #cmap = cm.hot
    #m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for vtx, p in zip(things, pressure):
        tri = a3.art3d.Poly3DCollection([vtx])
        #tri.set_color(m.to_rgba(p))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

    #fig.colorbar(m, ax=ax)
    ###
    # plot water level
    water_size = int(1.1*math.ceil(max(max(abs(leading_mid_points[:, 2])), max(abs(trailing_mid_points[:, 2])))))
    xx, yy = np.meshgrid(range(-water_size, water_size), range(-water_size, water_size))
    zz = 0 * xx + 0 * yy
    ax.plot_surface(xx, yy, zz, alpha=0.25)

    set_ax_eq(ax, xx, yy, zz)
    return ax, water_size

def display_hull(ax: plt.Axes, hull: HullGeometry):
    """
    display_hull plots hull 

    :param plt.Axes ax: axes for drawing
    :param HullGeometry hull: hull object
    """
    ax.plot(hull.deck_centerline[:, 0], hull.deck_centerline[:, 1], hull.deck_centerline[:, 2], 'gray')
    ax.plot(hull.deck_port_line[:, 0], hull.deck_port_line[:, 1], hull.deck_port_line[:, 2], 'gray')
    ax.plot(hull.deck_starboard_line[:, 0], hull.deck_starboard_line[:, 1], hull.deck_starboard_line[:, 2], 'gray')
    ax.plot(hull.deck_port_line_underwater[:, 0], hull.deck_port_line_underwater[:, 1], hull.deck_port_line_underwater[:, 2], 'gray', alpha=0.25)
    ax.plot(hull.deck_starboard_line_underwater[:, 0], hull.deck_starboard_line_underwater[:, 1], hull.deck_starboard_line_underwater[:, 2], 'gray', alpha=0.25)


def display_winds(ax : plt.Axes, cp_points : np.ndarray, water_size : int,  inlet_condition: InletConditions, inviscid_flow_results :  InviscidFlowResults, n_spanwise : int, n_chordwise : int, show_induced_wind: bool = False):
    """
    display_winds displays winds on final plot

    :param plt.Axes ax: ax
    :param np.ndarray cp_points: array with center of pressure points
    :param int water_size: size of water
    :param InletConditions inlet_condition: inlet conditions
    :param InviscidFlowResults inviscid_flow_results: flow results
    """
    #N = len(cp_points[:, 2])
     
    mean_AWA = np.mean(inlet_condition.AWA_infs_deg)
    shift_x = (-0.925) * water_size * np.cos(np.deg2rad(mean_AWA))
    shift_y = (-0.925) * water_size * np.sin(np.deg2rad(mean_AWA))

    #######
    # example:
    # cp_points has 24 points, n_spanwose = 2, n_chordise=3 it means that
    # we have: 24 / (2*3) = 24 / 6 = 4 elements: jib + main above water and jib and main under water
    l = int(cp_points.shape[0] / (n_spanwise * n_chordwise))
    # if you have main and jib this list should have 4 elements
    # if only jib - 2 elements
    
    ###
    #####
    # get winds acting only on first spanwise section on each sail
    tws_chunks = np.split(inlet_condition.tws_at_cp, n_chordwise*l)
    tws_chunks = np.asarray(tws_chunks)
    tws_chunks_at_first_spanwise_sections = tws_chunks[0::n_chordwise]
    sh0, sh1, _ = tws_chunks_at_first_spanwise_sections.shape
    tws_chunks_at_first_spanwise_sections = tws_chunks_at_first_spanwise_sections.reshape(sh0 * sh1 ,3)
    
    V_app_infs_chunks = np.split(inlet_condition.V_app_infs, n_chordwise*l)
    V_app_infs_chunks = np.asarray(V_app_infs_chunks)
    V_app_infs_at_first_spanwise_sections = V_app_infs_chunks[0::n_chordwise]
    sh0, sh1, _ = V_app_infs_at_first_spanwise_sections.shape
    V_app_infs_at_first_spanwise_sections = V_app_infs_at_first_spanwise_sections.reshape(sh0 * sh1 ,3)
    
    cp_points_chunks = np.split(inlet_condition.cp_points, n_chordwise*l)
    cp_points_chunks = np.asarray(cp_points_chunks)
    cp_points_at_first_spanwise_sections = cp_points_chunks[0::n_chordwise]
    sh0, sh1, _ = cp_points_at_first_spanwise_sections.shape
    cp_points_at_first_spanwise_sections = cp_points_at_first_spanwise_sections.reshape(sh0 * sh1 ,3)
    
    # assert
    V_winds = [tws_chunks_at_first_spanwise_sections, V_app_infs_at_first_spanwise_sections]
    colors = ['green', 'blue']  # G: True wind, B: - Apparent wind, R: Induced@CP
    if show_induced_wind:
        V_induced_at_cp_chunks = np.split(inviscid_flow_results.V_induced_at_cp, n_chordwise*l)
        V_induced_at_cp_chunks = np.asarray(V_induced_at_cp_chunks)
        V_induced_at_cp_chunks_at_first_spanwise_sections = V_induced_at_cp_chunks[0::n_chordwise]
        sh0, sh1, _ = V_induced_at_cp_chunks_at_first_spanwise_sections.shape
        V_induced_at_cp_chunks_at_first_spanwise_sections = V_induced_at_cp_chunks_at_first_spanwise_sections.reshape(sh0 * sh1 ,3)
        V_winds.append(V_induced_at_cp_chunks_at_first_spanwise_sections)
        colors.append('red')

    for arr in V_winds:
        assert arr.shape[0] == n_spanwise * l, 'Not proper array shape, check lines above!'

    app_induced_colors = ['peru', 'red', 'peru', 'red']
    if l == 2:
        app_induced_colors = ['peru', 'red']
    # check if colors are defined for appaernt + induced wind - above and under water
    assert (len(app_induced_colors) == l and len(app_induced_colors) % 2 == 0), "Bad length of app_induced_colors list"
    color_counter = 0
    for V_wind, color in zip(V_winds, colors):
        for i in range(n_spanwise*l):
            # for chnaging colors for induced wind based on sail (jib above, main above, jib_underwater, main_underwater etc) 
            draw_color = color
            if color == 'red':
                if i % n_spanwise == 0 and i !=0:
                    color_counter += 1
                draw_color = app_induced_colors[color_counter]
                    
                shift_x = cp_points_at_first_spanwise_sections[i, 0]
                shift_y = cp_points_at_first_spanwise_sections[i, 1]

            vx = np.array([shift_x, shift_x+V_wind[i, 0]])
            vy = np.array([shift_y, shift_y+V_wind[i, 1]])
            vz = np.array([cp_points_at_first_spanwise_sections[i, 2], cp_points_at_first_spanwise_sections[i, 2]])

            if cp_points_at_first_spanwise_sections[i, 2] > 0:
                arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color=draw_color, alpha=0.75)
            else:
                arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color=draw_color, alpha=0.05)
            ax.add_artist(arrow)


def display_CE_CLR(ax : plt.Axes,
                   inviscid_flow_results: InviscidFlowResults,
                   hull: HullGeometry) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    display_CE_CLR display force 

    :param plt.Axes ax: plot ax
    :param InviscidFlowResults inviscid_flow_results: flow results
    :param HullGeometry hull: hull object
    :return Tuple[float, np.ndarray, np.ndarray]: Tuple[scale, lateral resistance, effort, force]
    """

    # https://en.wikipedia.org/wiki/Forces_on_sails#Forces_on_sailing_craft
    def plot_vector(origin, length):
        vx = np.array([origin[0], origin[0] + length[0]])
        vy = np.array([origin[1], origin[1] + length[1]])
        vz = np.array([origin[2], origin[2] + length[2]])
        arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color='black', alpha=0.75)
        ax.add_artist(arrow)
        ax.scatter3D(origin[0], origin[1], origin[2], c='black', marker="o")

    scale = 0.8*np.mean(inviscid_flow_results.V_app_fs_length)  # ~10 gives nice plot
    clr = hull.center_of_lateral_resistance
    ce = inviscid_flow_results.above_water_centre_of_effort_estimate_xyz

    F = scale * normalize(inviscid_flow_results.F_xyz_total)
    plot_vector(ce, F)
    plot_vector(clr, -F)
    
    return scale, clr, ce, F

def display_panels_xyz_and_winds(vlm :Vlm, inviscid_flow_results: InviscidFlowResults, 
                                 inlet_condition: InletConditions,
                                 hull: HullGeometry,
                                 show_plot: bool = True, show_induced_wind: bool = False,
                                 is_sailopt_mode=False
                                 ):
    """
    display_panels_xyz_and_winds plot whole yacht with winds and force

    :param Vlm vlm: Vlm object
    :param InviscidFlowResults inviscid_flow_results: flow results
    :param InletConditions inlet_condition: inlet conditions
    :param HullGeometry hull: hull object
    :param bool show_plot: decide if plot should be shown, defaults to True
    """
    
    color_panels_by = None
    title_mode = None
    if is_sailopt_mode:
        color_panels_by = 'gamma_magnitude'
        title_mode = 'SailOpt mode: Panels colored by optimal circulation \n'
    else:
        color_panels_by = 'p_coeffs'
        title_mode = 'VLM mode: Panels colored by coefficient of pressure \n'


    ax, water_size = display_panels_xyz(vlm, color_panels_by)

    red_text = ''
    if show_induced_wind:
        red_text = ', Induced@CP (red)' 
    
    ax.set_title(f'{title_mode}'
                 f'Winds: True (green), Apparent (blue){red_text} \n'
                 'Centre of Effort & Center of Lateral Resistance (black)')
    
    
    display_hull(ax, hull)

    display_winds(ax, vlm.cp, water_size, inlet_condition, inviscid_flow_results, vlm.n_spanwise, vlm.n_chordwise, show_induced_wind)

    scale, clr, ce, F = display_CE_CLR(ax, inviscid_flow_results, hull)
    

    if show_plot:
        plt.show()


def plot_cp(mesh : np.ndarray, p_coeffs : np.ndarray, path_to_save : str):
    hv.renderer('bokeh')
    boundaries = sp.geometry.PolygonArray([[np.vstack([panel[:,[0,2]], panel[:,[0,2]][0]]).flatten()] for panel in mesh])
    info = sp.GeoDataFrame({'boundary': boundaries, 'p_coeffs': list(p_coeffs)}) 
    
    ropts = dict(tools=["hover"], height=380, width=330, colorbar=True, colorbar_position="right", color='p_coeffs')
    hvpolys = hv.Polygons(info, vdims=['p_coeffs']).opts(**ropts)
    
    try:
        __IPYTHON__
        from bokeh.io import show, output_notebook
        output_notebook()
        show(hv.render(hvpolys, backend='bokeh'))
    except NameError:
        hv.save(hvpolys, path_to_save + '/cp_plot.html', backend='bokeh')
        
def plot_section_coeff(x_data : list, y_data: list, file_name: str, coeff_name: str, colors : list):
    
    allowed_names = ('lift', 'drag')
    try:
        if coeff_name not in allowed_names:
            raise ValueError(f"Bad coeff_name value, allowed: {allowed_names}")
    except ValueError as err:
        print(err)
        sys.exit(1)
        
    plt.style.use(['science', 'grid'])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    l = len(y_data)
    lbs = ['jib', 'main']
    markers = ["o", "^"]
    
    c_name = 'cl'
    str_latex = '$c_l$'
    if coeff_name == 'drag':
        c_name = 'cd'
        str_latex = '$c_d$'
        
    for i in range(l):
        ax1.scatter(x_data[i], y_data[i], label=lbs[i], s=5, marker=markers[i], color=colors[i])

    plt.ylabel('Height [m]')
    plt.xlabel(fr'Section {str_latex}')
    plt.legend(loc='center left')
    ax1.autoscale(tight=True)
    
    try:
        __IPYTHON__
        plt.savefig(f'section_{c_name}', dpi=300)
        plt.show()
    except NameError:
        plt.savefig(file_name + '/' +  f'section_{c_name}', dpi=300)
    
