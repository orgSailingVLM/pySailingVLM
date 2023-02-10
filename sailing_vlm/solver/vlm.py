from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sailing_vlm.inlet.inlet_conditions import InletConditions
from sailing_vlm.solver import coefs
from sailing_vlm.solver import forces
from sailing_vlm.solver import velocity
from sailing_vlm.solver import panels
from typing import List
from typing import ClassVar
from sailing_vlm.solver.panels_plotter import display_panels_or_rings

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

# sprawdzic typy !!!!
@dataclass
class Vlm:


    panels: np.ndarray
    n_chordwise : int
    n_spanwise :int

    rho : float
    wind : np.ndarray
    trailing_edge_info : np.ndarray
    leading_edge_info : np.ndarray


    # post init atributes
    areas : np.ndarray = field(init=False)
    normals : np.ndarray = field(init=False)
    leading_mid_points : np.ndarray = field(init=False)
    trailing_mid_points : np.ndarray = field(init=False)
    collocation_points : np.ndarray = field(init=False)
    center_of_pressure : np.ndarray = field(init=False)
    rings : np.ndarray = field(init=False)
    span_vectors : np.ndarray = field(init=False)
    coefs : np.ndarray = field(init=False)
    RHS : np.ndarray = field(init=False)
    wind_coefs : np.ndarray = field(init=False)
    gamma_magnitude : np.ndarray = field(init=False)
    inlet_conditions: InletConditions = field(init=False)
    force : np.ndarray = field(init=False)
    pressure : np.ndarray = field(init=False)
    V_app_fs_at_cp : np.ndarray = field(init=False)
    V_induced_at_cp : np.ndarray = field(init=False)

    # as class var
    gamma_orientation : float = -1.0

    def __post_init__(self):

        # M = wzdluz rozpoetosci skrzydel, spanwise
        # N = chordwise, linia laczaca leading i trailing
        self.areas = panels.get_panels_area(self.panels)
        self.normals, self.collocation_points, self.center_of_pressure, self.rings, self.span_vectors, self.leading_mid_points, self.trailing_mid_points = coefs.calculate_normals_collocations_cps_rings_spans_leading_trailing_mid_points(self.panels, self.gamma_orientation)
        self.normals2, self.collocation_points2, self.center_of_pressure2, self.rings2, self.span_vectors2, self.leading_mid_points2, self.trailing_mid_points2 = coefs.calculate_stuff(self.panels, self.trailing_edge_info, self.gamma_orientation, self.n_chordwise, self.n_spanwise)
        
        np.testing.assert_almost_equal(self.normals, self.normals2)
        np.testing.assert_almost_equal(self.collocation_points, self.collocation_points2)
        np.testing.assert_almost_equal(self.center_of_pressure, self.center_of_pressure2)
        np.testing.assert_almost_equal(self.span_vectors, self.span_vectors2)
        np.testing.assert_almost_equal(self.leading_mid_points, self.leading_mid_points2)
        np.testing.assert_almost_equal(self.trailing_mid_points, self.trailing_mid_points2)
        
        # tylko rigi beda inne ale tak ma byc wiec assert nie przejdzie
        # np.testing.assert_almost_equal(self.rings, self.rings2)
        self.normals = self.normals2
        self.collocation_points = self.collocation_points2
        self.center_of_pressure = self.center_of_pressure2
        self.rings = self.rings2
        self.span_vectors = self.span_vectors2
        self.leading_mid_points = self.leading_mid_points2
        self.trailing_mid_points = self.trailing_mid_points2
  
        #self.__show_rings(self.rings2, self.n_spanwise, self.n_chordwise)

        self.inlet_conditions = InletConditions(self.wind, self.rho, self.center_of_pressure)

        self.coefs, self.RHS, self.wind_coefs = coefs.get_influence_coefficients_spanwise( self.collocation_points, self.rings, self.normals, self.inlet_conditions.V_app_infs, self.trailing_edge_info, self.gamma_orientation)
        self.gamma_magnitude = coefs.solve_eq( self.coefs,  self.RHS)

        self.V_induced_at_ctrl,  self.V_app_fs_at_ctrl_p = velocity.calculate_app_fs(self.inlet_conditions.V_app_infs,  self.wind_coefs,  self.gamma_magnitude)

        # boundary condition calculated in collocation points (control points)
        assert forces.is_no_flux_BC_satisfied(self.V_app_fs_at_ctrl_p, self.panels, self.areas, self.normals)
        self.force, self.V_app_fs_at_cp, self.V_induced_at_cp = forces.calc_force_wrapper(self.inlet_conditions.V_app_infs, self.gamma_magnitude, self.rho, self.center_of_pressure, self.rings, self.n_spanwise, self.normals, self.span_vectors, self.trailing_edge_info, self.leading_edge_info, self.gamma_orientation)

        self.pressure = forces.calc_pressure(self.force, self.normals, self.areas)


        display_panels_or_rings(self.rings2, self.pressure, self.leading_mid_points2, self.trailing_mid_points2)
        # comment this section if debugging camber
        plt.show()
        print()

    def __show_draft(self):
        sh0, sh1, sh2 = self.rings.shape
        # debugging
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # bo mamy jib i maon i odbicie
        l = int(self.panels.shape[0] / 4)

        xx = self.panels.reshape(sh0*sh1, sh2)[0:l].transpose()[0]
        yy = self.panels.reshape(sh0*sh1, sh2)[0:l].transpose()[1]
        zz = self.panels.reshape(sh0*sh1, sh2)[0:l].transpose()[2]

        xx1 = self.rings.reshape(sh0*sh1, sh2)[0:l].transpose()[0]
        yy1 = self.rings.reshape(sh0*sh1, sh2)[0:l].transpose()[1]
        zz1 = self.rings.reshape(sh0*sh1, sh2)[0:l].transpose()[2]

        x = np.concatenate((xx, xx1))
        y = np.concatenate((yy, yy1))
        z = np.concatenate((zz, zz1))
        x_min, x_max = np.amin(x), np.amax(x)
        y_min, y_max = np.amin(y), np.amax(y)
        z_min, z_max = np.amin(z), np.amax(z)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        # https://matplotlib.org/stable/gallery/images_contours_and_fields/quiver_demo.html
        # WHAAAT ???

        # "Note: The plot autoscaling does not take into account the arrows, so
        # those on the boundaries may reach out of the picture. This is not an easy
        # problem to solve in a perfectly general way. The recommended workaround is
        # to manually set the Axes limits
        # in such a case."
        ax.set_xlim3d(-5, 0)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(0, 11)

        colors = {0: 'r', 1: 'g', 2: 'b'}
        assert len(colors) == l
        for i in range(l):
            ax.plot(self.panels[i].transpose()[0], self.panels[i].transpose()[1], self.panels[i].transpose()[2], f'x{colors[i]}-')
            ax.plot(self.rings[i].transpose()[0], self.rings[i].transpose()[1], self.rings[i].transpose()[2], f'.{colors[i]}--')
            ax.quiver(self.collocation_points[i][0], self.collocation_points[i][1], self.collocation_points[i][2], self.normals[i][0], self.normals[i][1], self.normals[i][2],  color=f'{colors[i]}', normalize=True, length=1)

        ### end of debugging
        plt.show()

    def __show_rings(self, rings_to_show : np.ndarray, n_spanwise : int, n_chordwise : int):

        G = n_spanwise * n_chordwise
        div = int(rings_to_show.shape[0] / G)
        splitted = np.array_split(rings_to_show, div)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        c = ['violet', 'purple', 'blue', 'slateblue', 'lightseagreen', 'turquoise']
        for rings in splitted:
            for i in range(G):
                
                x = rings[i].transpose()[0].tolist()
                y = rings[i].transpose()[1].tolist()
                z = rings[i].transpose()[2].tolist()
                
                x.append(x[0])
                y.append(y[0])
                z.append(z[0])
                ax.scatter(x,y,z, c='red',s=100)
                ax.plot(x,y,z, color='red')
            
                #ax.scatter(x,y,z, c=c[i],s=100)
                #ax.plot(x,y,z, color=c[i])
            
        plt.show()

        print()


