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
        
        # ax.set_xlim3d(-5, 8)
        # ax.set_ylim3d(-5, 8)
        # ax.set_zlim3d(-5, 8)
    

        # l1 = x_max - x_min
        # l2 = y_max - y_min
        # l3 = z_max - z_min
        
        # ax.set_xlim3d(x_min - l1, x_max + l1)
        # ax.set_ylim3d(y_min - l2 , y_max + l2)
        # ax.set_zlim3d(z_min - l3 , z_max + l3)
        
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
        self.inlet_conditions = InletConditions(self.wind, self.rho, self.center_of_pressure)
        
        self.coefs, self.RHS, self.wind_coefs = coefs.get_influence_coefficients_spanwise( self.collocation_points, self.rings, self.normals, self.inlet_conditions.V_app_infs, self.trailing_edge_info, self.gamma_orientation)
        self.gamma_magnitude = coefs.solve_eq( self.coefs,  self.RHS)

        self.V_induced_at_ctrl,  self.V_app_fs_at_ctrl_p = velocity.calculate_app_fs(self.inlet_conditions.V_app_infs,  self.wind_coefs,  self.gamma_magnitude)

        assert forces.is_no_flux_BC_satisfied(self.V_app_fs_at_ctrl_p, self.panels, self.areas, self.normals)
        
        self.force, self.V_app_fs_at_cp, self.V_induced_at_cp = forces.calc_force_wrapper(self.inlet_conditions.V_app_infs, self.gamma_magnitude, self.rho, self.center_of_pressure, self.rings, self.n_spanwise, self.normals, self.span_vectors, self.trailing_edge_info, self.leading_edge_info, self.gamma_orientation)

        self.pressure = forces.calc_pressure(self.force, self.normals, self.areas)

        # comment this section if debugging camber
        
        
        # np.savetxt('trailing_edge_info.txt', self.trailing_edge_info)
        # np.savetxt('leading_edge_info.txt', self.leading_edge_info)
        # np.savetxt('V_app_infs.txt', self.inlet_conditions.V_app_infs)
        # np.savetxt('areas.txt', self.areas)
        # np.savetxt('normals.txt', self.normals) # decimal 4
        # np.savetxt('collocation_points.txt', self.collocation_points) # dec 4
        # np.savetxt('center_of_pressure.txt', self.center_of_pressure) # dec 4
        # np.savetxt('rings.txt', self.rings.reshape(sh0*sh1,sh2)) # dec 4
        # np.savetxt('span_vectors.txt', self.span_vectors) # dec 5
        # np.savetxt('leading_mid_points.txt', self.leading_mid_points) # dec 4
        # np.savetxt('trailing_mid_points.txt', self.trailing_mid_points) # dec 4
        # np.savetxt('coefs.txt', self.coefs) # dec 5
        # np.savetxt('RHS.txt', self.RHS) # dec 3
        # np.savetxt('wind_coefs.txt', self.wind_coefs.reshape(sh0*sh0,sh2)) # dec 4
        # np.savetxt('gamma_magnitude.txt', self.gamma_magnitude) # dec 3
        # np.savetxt('V_induced_at_ctrl.txt', self.V_induced_at_ctrl) # dec 3
        # np.savetxt('V_app_fs_at_ctrl_p.txt', self.V_app_fs_at_ctrl_p) # dec 3
        # np.savetxt('V_app_fs_at_cp.txt', self.V_app_fs_at_cp) # podejrzany!!!!
        # np.savetxt('V_induced_at_cp.txt', self.V_induced_at_cp)
        # np.savetxt('force.txt', self.force)
        # np.savetxt('pressure.txt', self.pressure)
        ### end of section
        
        # np.testing.assert_almost_equal(np.loadtxt('leading_edge_info.txt'), self.leading_edge_info)
        # np.testing.assert_almost_equal(np.loadtxt('trailing_edge_info.txt'), self.trailing_edge_info)
        # np.testing.assert_almost_equal(np.loadtxt('V_app_infs.txt'), self.inlet_conditions.V_app_infs)
        # np.testing.assert_almost_equal(np.loadtxt('areas.txt'), self.areas)
        # np.testing.assert_almost_equal(np.loadtxt('normals.txt'), self.normals, decimal=6)
        # np.testing.assert_almost_equal(np.loadtxt('collocation_points.txt'), self.collocation_points, decimal=6)
        # np.testing.assert_almost_equal(np.loadtxt('center_of_pressure.txt'), self.center_of_pressure, decimal=6)
        # np.testing.assert_almost_equal(np.loadtxt('rings.txt'), self.rings.reshape(sh0*sh1,sh2), decimal=6)
        # np.testing.assert_almost_equal(np.loadtxt('span_vectors.txt'), self.span_vectors, decimal=6)
        # np.testing.assert_almost_equal(np.loadtxt('leading_mid_points.txt'), self.leading_mid_points, decimal=6)
        # np.testing.assert_almost_equal(np.loadtxt('trailing_mid_points.txt'), self.trailing_mid_points, decimal=6)
        # np.testing.assert_almost_equal(np.loadtxt('coefs.txt'), self.coefs)
        # np.testing.assert_almost_equal(np.loadtxt('RHS.txt'), self.RHS, decimal=5)
        # np.testing.assert_almost_equal(np.loadtxt('wind_coefs.txt'), self.wind_coefs.reshape(sh0*sh0,sh2), decimal=6)
        # np.testing.assert_almost_equal(np.loadtxt('gamma_magnitude.txt'), self.gamma_magnitude, decimal=5)
        # np.testing.assert_almost_equal(np.loadtxt('V_induced_at_ctrl.txt'), self.V_induced_at_ctrl, decimal=5)
        # np.testing.assert_almost_equal(np.loadtxt('V_app_fs_at_ctrl_p.txt'), self.V_app_fs_at_ctrl_p, decimal=5)
        # np.testing.assert_almost_equal(np.loadtxt('V_app_fs_at_cp.txt'), self.V_app_fs_at_cp, decimal=6) # !!!!!!!! max absolute difference: 67642717.9454928
        # np.testing.assert_almost_equal(np.loadtxt('V_induced_at_cp.txt'), self.V_induced_at_cp, decimal=6) ### ax absolute difference: 67642717.94549279
        # np.testing.assert_almost_equal(np.loadtxt('force.txt'), self.force, decimal=3)
        # np.testing.assert_almost_equal(np.loadtxt('pressure.txt'), self.pressure, decimal=3)
        

