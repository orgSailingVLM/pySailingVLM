
import numpy as np
import math

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sailingVLM.NewApproach.vlm import NewVlm

from sailingVLM.Solver.vortices import normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl

from sailingVLM.ResultsContainers.InviscidFlowResults import InviscidFlowResults
from sailingVLM.Inlet.InletConditions import InletConditions
from sailingVLM.YachtGeometry.HullGeometry import HullGeometry


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


def _prepare_geometry_data_to_display(panels1d):
    le_mid_points = np.array([panel.get_leading_edge_mid_point() for panel in panels1d])
    cp_points = np.array([panel.cp_position for panel in panels1d])
    ctr_points = np.array([panel.get_ctr_point_position() for panel in panels1d])
    te_midpoints = np.array([panel.get_trailing_edge_mid_points() for panel in panels1d])

    return le_mid_points, cp_points, ctr_points, te_midpoints


def display_panels_xyz(panels1d):

    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    # ax.set_title('Initial location of: \n Leading Edge, Lifting Line, Control Points and Trailing Edge')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def set_ax_eq(ax, X, Y, Z):
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

    # Data for a three-dimensional line
    le_mid_points, cp_points, ctr_points, te_mid_points = _prepare_geometry_data_to_display(panels1d)
    # ax.scatter3D(le_mid_points[:, 0], le_mid_points[:, 1], le_mid_points[:, 2], c=le_mid_points[:, 2], marker="<", cmap='Greys')
    # ax.scatter3D(cp_points[:, 0], cp_points[:, 1], cp_points[:, 2], c=cp_points[:, 2], marker="o", cmap='Greens', s=2)  # s stands for size
    # ax.scatter3D(ctr_points[:, 0], ctr_points[:, 1], ctr_points[:, 2], c=ctr_points[:, 2], marker="x", cmap='Blues')
    # ax.scatter3D(te_mid_points[:, 0], te_mid_points[:, 1], te_mid_points[:, 2], c=te_mid_points[:, 2], marker=">", cmap='Greys')

    ### plot panels and color by pressure
    # https://stackoverflow.com/questions/15140072/how-to-map-number-to-color-using-matplotlibs-colormap

    pressures = np.array([panel.pressure for panel in panels1d])
    norm = mpl.colors.Normalize(vmin=min(pressures), vmax=max(pressures))
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for panel in panels1d:
        vtx = panel.get_points()
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(m.to_rgba(panel.pressure))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

    fig.colorbar(m, ax=ax)
    ###
    # plot water level
    water_size = int(1.1*math.ceil(max(max(abs(le_mid_points[:, 2])), max(abs(te_mid_points[:, 2])))))
    xx, yy = np.meshgrid(range(-water_size, water_size), range(-water_size, water_size))
    zz = 0 * xx + 0 * yy
    ax.plot_surface(xx, yy, zz, alpha=0.25)

    set_ax_eq(ax, xx, yy, zz)
    return ax, cp_points, water_size


def display_panels_xyz_new_approach(myvlm):
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    # ax.set_title('Initial location of: \n Leading Edge, Lifting Line, Control Points and Trailing Edge')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def set_ax_eq(ax, X, Y, Z):
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

    # Data for a three-dimensional line
    
    # ax.scatter3D(le_mid_points[:, 0], le_mid_points[:, 1], le_mid_points[:, 2], c=le_mid_points[:, 2], marker="<", cmap='Greys')
    # ax.scatter3D(cp_points[:, 0], cp_points[:, 1], cp_points[:, 2], c=cp_points[:, 2], marker="o", cmap='Greens', s=2)  # s stands for size
    # ax.scatter3D(ctr_points[:, 0], ctr_points[:, 1], ctr_points[:, 2], c=ctr_points[:, 2], marker="x", cmap='Blues')
    # ax.scatter3D(te_mid_points[:, 0], te_mid_points[:, 1], te_mid_points[:, 2], c=te_mid_points[:, 2], marker=">", cmap='Greys')

    ### plot panels and color by pressure
    # https://stackoverflow.com/questions/15140072/how-to-map-number-to-color-using-matplotlibs-colormap
    #panel_points = np.array([panel.get_points() for panel in myvlm])
    norm = mpl.colors.Normalize(vmin=min(myvlm.pressure), vmax=max(myvlm.pressure))
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for vtx, p in zip(myvlm.panels, myvlm.pressure):
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(m.to_rgba(p))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

    fig.colorbar(m, ax=ax)
    ###
    # plot water level
    water_size = int(1.1*math.ceil(max(max(abs(myvlm.leading_mid_points[:, 2])), max(abs(myvlm.trailing_mid_points[:, 2])))))
    xx, yy = np.meshgrid(range(-water_size, water_size), range(-water_size, water_size))
    zz = 0 * xx + 0 * yy
    ax.plot_surface(xx, yy, zz, alpha=0.25)

    set_ax_eq(ax, xx, yy, zz)
    return ax, myvlm.center_of_pressure, water_size



def display_hull(ax, hull: HullGeometry):
    ax.plot(hull.deck_centerline[:, 0], hull.deck_centerline[:, 1], hull.deck_centerline[:, 2], 'gray')
    ax.plot(hull.deck_port_line[:, 0], hull.deck_port_line[:, 1], hull.deck_port_line[:, 2], 'gray')
    ax.plot(hull.deck_starboard_line[:, 0], hull.deck_starboard_line[:, 1], hull.deck_starboard_line[:, 2], 'gray')
    ax.plot(hull.deck_port_line_underwater[:, 0], hull.deck_port_line_underwater[:, 1], hull.deck_port_line_underwater[:, 2], 'gray', alpha=0.25)
    ax.plot(hull.deck_starboard_line_underwater[:, 0], hull.deck_starboard_line_underwater[:, 1], hull.deck_starboard_line_underwater[:, 2], 'gray', alpha=0.25)


def display_winds(ax, cp_points, water_size,  inlet_condition: InletConditions, inviscid_flow_results):
    N = len(cp_points[:, 2])

    
    mean_AWA = np.mean(inlet_condition.AWA_infs_deg)
    shift_x = (-0.925) * water_size * np.cos(np.deg2rad(mean_AWA))
    shift_y = (-0.925) * water_size * np.sin(np.deg2rad(mean_AWA))

    V_winds = [inlet_condition.tws_at_cp, inlet_condition.V_app_infs, inviscid_flow_results.V_app_fs_at_cp]
    colors = ['green', 'blue', 'red']  # G: True wind, B: - Apparent wind, R: Apparent + Induced wind
   
    zipp = zip(V_winds, colors)
    for V_wind, color in zip(V_winds, colors):
        # V_wind = V_winds[2]
        # color = colors[2]
        for i in range(N):
            # vx = np.array([cp_points[i, 0], cp_points[i, 0] + V_wind[i, 0]])
            # vy = np.array([cp_points[i, 1], cp_points[i, 1] + V_wind[i, 1]])
            vx = np.array([shift_x, shift_x+V_wind[i, 0]])
            vy = np.array([shift_y, shift_y+V_wind[i, 1]])
            vz = np.array([cp_points[i, 2], cp_points[i, 2]])

            # ax.plot(vx, vy, vz,color='red', alpha=0.8, lw=1)  # old way
            # arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>",  c=cp_points[:, 2], cmap='Greys')
            if cp_points[i, 2] > 0:
                arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color=color, alpha=0.75)
            else:
                arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color=color, alpha=0.15)
            ax.add_artist(arrow)
    # potem wywalic tego returna
    return V_winds


def display_winds_new_approach(ax, cp_points, water_size,  inlet_condition: InletConditions, inviscid_flow_results):
    N = len(cp_points[:, 2])

    
    mean_AWA = np.mean(inlet_condition.AWA_infs_deg)
    shift_x = (-0.925) * water_size * np.cos(np.deg2rad(mean_AWA))
    shift_y = (-0.925) * water_size * np.sin(np.deg2rad(mean_AWA))

    V_winds = [inlet_condition.tws_at_cp, inlet_condition.V_app_infs, inviscid_flow_results.V_app_fs_at_cp]
    colors = ['green', 'blue', 'red']  # G: True wind, B: - Apparent wind, R: Apparent + Induced wind
    zipp = zip(V_winds, colors)
    for V_wind, color in zip(V_winds, colors):
        # V_wind = V_winds[2]
        # color = colors[2]
        for i in range(N):
            # vx = np.array([cp_points[i, 0], cp_points[i, 0] + V_wind[i, 0]])
            # vy = np.array([cp_points[i, 1], cp_points[i, 1] + V_wind[i, 1]])
            vx = np.array([shift_x, shift_x+V_wind[i, 0]])
            vy = np.array([shift_y, shift_y+V_wind[i, 1]])
            vz = np.array([cp_points[i, 2], cp_points[i, 2]])

            # ax.plot(vx, vy, vz,color='red', alpha=0.8, lw=1)  # old way
            # arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>",  c=cp_points[:, 2], cmap='Greys')
            if cp_points[i, 2] > 0:
                arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color=color, alpha=0.75)
            else:
                arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color=color, alpha=0.15)
            ax.add_artist(arrow)
    return V_winds, ax, zipp


    
def display_CE_CLR(ax,
                   inviscid_flow_results: InviscidFlowResults,
                   hull: HullGeometry):
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

def display_CE_CLR_new_approach(ax,
                   myvlm : NewVlm,
                   hull: HullGeometry):
    # https://en.wikipedia.org/wiki/Forces_on_sails#Forces_on_sailing_craft
    def plot_vector(origin, length):
        vx = np.array([origin[0], origin[0] + length[0]])
        vy = np.array([origin[1], origin[1] + length[1]])
        vz = np.array([origin[2], origin[2] + length[2]])
        arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color='black', alpha=0.75)
        ax.add_artist(arrow)
        ax.scatter3D(origin[0], origin[1], origin[2], c='black', marker="o")

    scale = 0.8*np.mean(myvlm.V_app_fs_length)  # ~10 gives nice plot
    clr = hull.center_of_lateral_resistance
    ce = myvlm.above_water_centre_of_effort_estimate_xyz

    F = scale * normalize(myvlm.F_xyz_total)
    plot_vector(ce, F)
    plot_vector(clr, -F)
    
def display_forces_xyz(ax, panels1d, inviscid_flow_results: InviscidFlowResults):
    scale = 0.2*np.mean(inviscid_flow_results.F_xyz_total)  # ~0.2 gives nice plot
    F_length = np.linalg.norm(inviscid_flow_results.F_xyz, axis=1)
    F_max = max(F_length)

    def plot_vector(origin, length):
        vx = np.array([origin[0], origin[0] + length[0]])
        vy = np.array([origin[1], origin[1] + length[1]])
        vz = np.array([origin[2], origin[2] + length[2]])
        arrow = Arrow3D(vx, vy, vz, mutation_scale=10, lw=1, arrowstyle="-|>", color='gray', alpha=0.75)
        ax.add_artist(arrow)
        ax.scatter3D(origin[0], origin[1], origin[2], c='black', marker="o")

    cp_points = np.array([panel.cp_position for panel in panels1d])
    for F, cp in zip(inviscid_flow_results.F_xyz, cp_points):
        F_normalized = scale*F/F_max
        plot_vector(cp, F_normalized)


def display_panels_xyz_and_winds_new_approach(myvlm : NewVlm,
                                 inviscid_flow_results: InviscidFlowResults,
                                 inlet_condition: InletConditions,
                                 hull: HullGeometry,
                                 show_plot=True

                                 ):
    ax, cp_points, water_size = display_panels_xyz(panels1d)
    ax.set_title('Panels colored by pressure \n'
                 'Winds: True (green), Apparent (blue), Apparent + Induced (red) \n'
                 'Centre of Effort & Center of Lateral Resistance (black)')

    display_hull(ax, hull)
    _, _, _ = display_winds_new_approach(ax, cp_points, water_size, inlet_condition, inviscid_flow_results)
    # display_forces_xyz(ax, panels1d, inviscid_flow_results)
    display_CE_CLR(ax, inviscid_flow_results, hull)

    if show_plot:
        plt.show()
        
        
def display_panels_xyz_and_winds(myvlm, inviscid_flow_results_new: InviscidFlowResults, panels1d,
                                 inlet_condition: InletConditions,
                                 my_inlet_condition: InletConditions,
                                 inviscid_flow_results: InviscidFlowResults,
                                 hull: HullGeometry,
                                 show_plot=True

                                 ):
    ax, cp_points, water_size = display_panels_xyz(panels1d)
    #ax, cp_points, water_size = display_panels_xyz(panels1d, inviscid_flow_results.pressure)
    my_ax, my_cp_points, my_water_size = display_panels_xyz_new_approach(myvlm)

    np.testing.assert_almost_equal(np.sort(cp_points, axis=0), np.sort(my_cp_points, axis=0))
    np.testing.assert_equal(water_size,my_water_size)
    ax.set_title('Panels colored by pressure \n'
                 'Winds: True (green), Apparent (blue), Apparent + Induced (red) \n'
                 'Centre of Effort & Center of Lateral Resistance (black)')
    
    my_ax.set_title('New approach')


    display_hull(ax, hull)
    display_hull(my_ax, hull)
    V_winds = display_winds(ax, cp_points, water_size, inlet_condition, inviscid_flow_results)
    my_V_winds = display_winds(my_ax, my_cp_points, my_water_size, my_inlet_condition, inviscid_flow_results_new)

    

    my_V_winds = np.asarray(my_V_winds).flatten()
    V_winds = np.asarray(V_winds).flatten()
    np.testing.assert_almost_equal(np.sort(V_winds, axis=0), np.sort(my_V_winds, axis=0))
    
    scale, clr, ce, F = display_CE_CLR(ax, inviscid_flow_results, hull)
    my_scale, my_clr, my_ce, my_F = display_CE_CLR(my_ax, inviscid_flow_results_new, hull)
    
    
    np.testing.assert_almost_equal(scale, my_scale)
    np.testing.assert_almost_equal(np.sort(clr, axis=0), np.sort(my_clr, axis=0))
    np.testing.assert_almost_equal(np.sort(ce, axis=0), np.sort(my_ce, axis=0))
    np.testing.assert_almost_equal(np.sort(F, axis=0), np.sort(my_F, axis=0))
    
    if show_plot:
        plt.show()

def display_panels_xz(panels1d):
    fig_name = f'plots/xy_initial_geometry_plot.png'
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(14, 8))
    plt.gca().set_aspect('equal', adjustable='box')

    axes = plt.gca()
    le_mid_points, cp_points, ctr_points, te_mid_points = _prepare_geometry_data_to_display(panels1d)

    plt.plot(le_mid_points[:, 0], le_mid_points[:, 2],
             color="grey", marker="<", markevery=1, markersize=7, linestyle="", linewidth=2,
             label=f'leading edge')

    plt.plot(cp_points[:, 0], cp_points[:, 2],
             color="green", marker="o", markevery=1, markersize=7, linestyle="", linewidth=2,
             label=f'cp_points')

    # plt.plot(ctr_points[:, 0], ctr_points[:, 2],
    #          color="blue", marker="x", markevery=1, markersize=7, linestyle="", linewidth=2,
    #          label=f'ctr_points')

    plt.plot(te_mid_points[:, 0], te_mid_points[:, 2],
             color="grey", marker=">", markevery=1, markersize=7, linestyle="", linewidth=2,
             label=f'trailing edge')

    # ------ format y axis ------ #
    # yll = ctr_points[upper_half:, 2].min()
    # yhl = ctr_points[upper_half:, 2].max()

    # yll = 0
    # yhl = max(ctr_points[:, 2])
    # axes.set_ylim([yll, yhl])
    # axes.set_yticks(np.linspace(yll, yhl, 11))

    # axes.set_yticks(np.arange(yll, yhl, 1E-2))
    # axes.set_yticks([1E-4, 1E-6, 1E-8, 1E-10, 1E-12])
    # axes.yaxis.set_major_formatter(xfmt)

    # plt.yscale('log')

    # ------ format x axis ------ #
    x_limits = int(2 * math.ceil(max(max(abs(le_mid_points[:, 0])), max(abs(te_mid_points[:, 0])))))
    plt.xlim(-x_limits, x_limits)

    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # scilimits=(-8, 8)

    plt.xlabel(f'X')
    plt.title(f'Location of: \n Leading Edge, Lifting Line and Trailing Edge')

    plt.ylabel(r'$Height$')
    plt.legend()
    plt.grid()

    fig = plt.gcf()  # get current figure
    # fig.savefig(fig_name, bbox_inches='tight')
    plt.show()
