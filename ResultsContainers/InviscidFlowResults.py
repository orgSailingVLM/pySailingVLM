import numpy as np
import pandas as pd
from Solver.forces import calc_moments, extract_above_water_quantities, calc_moment_arm_in_shifted_csys
from Solver.forces import determine_vector_from_its_dot_and_cross_product
from Rotations.CSYS_transformations import CSYS_transformations
from YachtGeometry.SailGeometry import SailSet

from Inlet.InletConditions import InletConditions
from Solver.forces import calc_force_LLT_xyz, calc_forces_on_panels_VLM_xyz


def prepare_inviscid_flow_results_llt(V_app_fs_at_cp, V_induced_at_cp, gamma_magnitude,
                                      sail_set: SailSet,
                                      inletConditions: InletConditions,
                                      csys_transformations: CSYS_transformations):

    # be careful, V_app_fs shall be calculated with respect to cp
    calc_force_LLT_xyz(V_app_fs_at_cp, gamma_magnitude, sail_set.panels1d, inletConditions.rho)
    [panel.calc_pressure() for panel in sail_set.panels1d]
    inviscid_flow_results = InviscidFlowResults(gamma_magnitude, V_induced_at_cp, V_app_fs_at_cp, sail_set, csys_transformations)
    return inviscid_flow_results

def prepare_inviscid_flow_results_vlm(gamma_magnitude,
                                      sail_set: SailSet,
                                      inlet_condition: InletConditions,
                                      csys_transformations: CSYS_transformations):

    calc_forces_on_panels_VLM_xyz(inlet_condition.V_app_infs, gamma_magnitude,
                                  sail_set.panels, inlet_condition.rho)

    [panel.calc_pressure() for panel in sail_set.panels1d]

    N = len(sail_set.panels1d)
    V_induced_at_cp = sail_set.V_induced_at_cp.reshape(N, 3)  # todo: get stuff from panels
    V_app_fs_at_cp = sail_set.V_app_fs_at_cp.reshape(N, 3)
    inviscid_flow_results = InviscidFlowResults(gamma_magnitude, V_induced_at_cp, V_app_fs_at_cp,
                                                sail_set, csys_transformations)
    return inviscid_flow_results


class InviscidFlowResults:
    def __init__(self, gamma_magnitude, V_induced_at_cp, V_app_fs_at_cp,
                 sail_set: SailSet,
                 csys_transformations: CSYS_transformations):

        cp_points = sail_set.get_cp_points1d()

        self.csys_transformations = csys_transformations
        self.gamma_magnitude = gamma_magnitude
        self.pressure = sail_set.pressures.flatten()
        self.V_induced_at_cp = V_induced_at_cp
        self.V_app_fs_at_cp = V_app_fs_at_cp

        self.V_induced_length = np.linalg.norm(self.V_induced_at_cp, axis=1)
        self.V_app_fs_length = np.linalg.norm(self.V_app_fs_at_cp, axis=1)
        self.AWA_app_fs = np.arctan(self.V_app_fs_at_cp[:, 1] / self.V_app_fs_at_cp[:, 0])
        # self.alfa_ind = alfa_app_infs - self.alfa_app_fs

        self.F_xyz = sail_set.forces_xyz.reshape(len(sail_set.panels1d), 3)  # todo: this may cause bugs when changing vstack/hstack arragment of panels in SailGeometry.py
        F_xyz_above_water, self.F_xyz_total = extract_above_water_quantities(self.F_xyz, cp_points)

        r = calc_moment_arm_in_shifted_csys(cp_points, csys_transformations.v_from_original_xyz_2_reference_csys_xyz)
        r_above_water, _ = extract_above_water_quantities(r, cp_points)

        dyn_dict = {}
        for i in range(len(sail_set.sails)):
            F_xyz_above_water_tmp = sail_set.extract_data_above_water_by_id(self.F_xyz, i)
            r_tmp = sail_set.extract_data_above_water_by_id(r, i)

            F_xyz_above_water_tmp_total = np.sum(F_xyz_above_water_tmp, axis=0)
            dyn_dict[f"F_{sail_set.sails[i].name}_total_COG.x"] = F_xyz_above_water_tmp_total[0]
            dyn_dict[f"F_{sail_set.sails[i].name}_total_COG.y"] = F_xyz_above_water_tmp_total[1]
            dyn_dict[f"F_{sail_set.sails[i].name}_total_COG.z"] = F_xyz_above_water_tmp_total[2]

            M_xyz_tmp = calc_moments(r_tmp, F_xyz_above_water_tmp)
            M_xyz_tmp_total = np.sum(M_xyz_tmp, axis=0)
            dyn_dict[f"M_{sail_set.sails[i].name}_total_COG.x"] = M_xyz_tmp_total[0]
            dyn_dict[f"M_{sail_set.sails[i].name}_total_COG.y"] = M_xyz_tmp_total[1]
            dyn_dict[f"M_{sail_set.sails[i].name}_total_COG.z"] = M_xyz_tmp_total[2]

        self.dyn_dict = dyn_dict

        self.M_xyz = calc_moments(r, self.F_xyz)
        _, self.M_total_above_water_in_xyz_csys = extract_above_water_quantities(self.M_xyz, cp_points)

        r_dot_F = np.array([np.dot(r[i], self.F_xyz[i]) for i in range(len(self.F_xyz))])
        _, r_dot_F_total_above_water = extract_above_water_quantities(r_dot_F, cp_points)

        self.above_water_centre_of_effort_estimate_xyz \
            = determine_vector_from_its_dot_and_cross_product(
            self.F_xyz_total, r_dot_F_total_above_water, self.M_total_above_water_in_xyz_csys)

        # r0 = self.M_total_above_water_in_xyz_csys[0] /self.F_xyz_total[0]
        # r1 = self.M_total_above_water_in_xyz_csys[1] / self.F_xyz_total[1]
        # r2 = self.M_total_above_water_in_xyz_csys[2] / self.F_xyz_total[2]
        # r_naive = np.array([r0,r1,r2])
        # np.cross(r_naive, self.F_xyz_total)

        self.F_centerline = csys_transformations.from_xyz_to_centerline_csys(self.F_xyz)
        _, self.F_centerline_total = extract_above_water_quantities(self.F_centerline, cp_points)

        self.M_centerline_csys = csys_transformations.from_xyz_to_centerline_csys(self.M_xyz)
        _, M_total_above_water_in_centerline_csys = extract_above_water_quantities(self.M_centerline_csys, cp_points)
        self.M_total_above_water_in_centerline_csys = M_total_above_water_in_centerline_csys

    def estimate_heeling_moment_from_keel(self, underwater_centre_of_effort_xyz):
        self.M_xyz_underwater_estimate_total = np.cross(underwater_centre_of_effort_xyz, -self.F_xyz_total)
        self.M_centerline_csys_underwater_estimate_total = \
        self.csys_transformations.from_xyz_to_centerline_csys([self.M_xyz_underwater_estimate_total])[0]

    def __iter__(self):
        yield 'Circulation_magnitude', self.gamma_magnitude
        yield 'V_induced_COG.x', self.V_induced_at_cp[:, 0]
        yield 'V_induced_COG.y', self.V_induced_at_cp[:, 1]
        yield 'V_induced_COG.z', self.V_induced_at_cp[:, 2]
        yield 'V_induced_length', self.V_induced_length
        yield 'V_app_fs_COG.x', self.V_app_fs_at_cp[:, 0]
        yield 'V_app_fs_COG.y', self.V_app_fs_at_cp[:, 1]
        yield 'V_app_fs_COG.z', self.V_app_fs_at_cp[:, 2]
        yield 'V_app_fs_length', self.V_app_fs_length
        yield 'AWA_fs_COG_deg', np.rad2deg(self.AWA_app_fs)
        yield 'AWA_fs_COW_deg', np.rad2deg(self.AWA_app_fs) - self.csys_transformations.leeway_deg
        yield 'Pressure', self.pressure
        yield 'F_sails_COG.x', self.F_xyz[:, 0]
        yield 'F_sails_COG.y', self.F_xyz[:, 1]
        yield 'F_sails_COG.z', self.F_xyz[:, 2]
        yield 'M_sails_COG.x', self.M_xyz[:, 0]
        yield 'M_sails_COG.y', self.M_xyz[:, 1]
        yield 'M_sails_COG.z', self.M_xyz[:, 2]
        yield 'F_sails_COW.x', self.F_centerline[:, 0]
        yield 'F_sails_COW.y', self.F_centerline[:, 1]
        yield 'F_sails_COW.z', self.F_centerline[:, 2]

    def to_df_full(self):
        obj_as_dict = dict(self)  # this calls __iter__
        df = pd.DataFrame.from_records(obj_as_dict)
        return df

    def to_df_integral(self):
        obj_as_dict = {
            'F_sails_total_COG.x': self.F_xyz_total[0],
            'F_sails_total_COG.y': self.F_xyz_total[1],
            'F_sails_total_COG.z': self.F_xyz_total[2],
            'F_sails_total_COW.x': self.F_centerline_total[0],
            'F_sails_total_COW.y': self.F_centerline_total[1],
            'F_sails_total_COW.z': self.F_centerline_total[2],
            'M_sails_total_COG.x': self.M_total_above_water_in_xyz_csys[0],
            'M_sails_total_COG.y': self.M_total_above_water_in_xyz_csys[1],
            'M_sails_total_COG.z': self.M_total_above_water_in_xyz_csys[2],
            'M_sails_total_COW.x (heel)': self.M_total_above_water_in_centerline_csys[0],
            'M_sails_total_COW.y (pitch)': self.M_total_above_water_in_centerline_csys[1],
            'M_sails_total_COW.z (yaw)': self.M_total_above_water_in_centerline_csys[2],
            'M_sails_total_COW.z (yaw - JG sign)': -self.M_total_above_water_in_centerline_csys[2],
        }

        if hasattr(self, 'M_xyz_underwater_estimate_total'):
            tmp = {
                'M_keel_total_COG.x': self.M_xyz_underwater_estimate_total[0],
                'M_keel_total_COG.y': self.M_xyz_underwater_estimate_total[1],
                'M_keel_total_COG.z': self.M_xyz_underwater_estimate_total[2],
                'M_keel_total_COW.x (heel)': self.M_centerline_csys_underwater_estimate_total[0],
                'M_keel_total_COW.y (pitch)': self.M_centerline_csys_underwater_estimate_total[1],
                'M_keel_total_COW.z (yaw)': self.M_centerline_csys_underwater_estimate_total[2],
                'M_keel_total_COW.z (yaw - JG sign)': -self.M_centerline_csys_underwater_estimate_total[2],
                'M_total_COG.x': self.M_total_above_water_in_xyz_csys[0] + self.M_xyz_underwater_estimate_total[0],
                'M_total_COG.y': self.M_total_above_water_in_xyz_csys[1] + self.M_xyz_underwater_estimate_total[1],
                'M_total_COG.z': self.M_total_above_water_in_xyz_csys[2] + self.M_xyz_underwater_estimate_total[2],
                'M_total_COW.x (heel)': self.M_total_above_water_in_centerline_csys[0] +
                                        self.M_centerline_csys_underwater_estimate_total[0],
                'M_total_COW.y (pitch)': self.M_total_above_water_in_centerline_csys[1] +
                                         self.M_centerline_csys_underwater_estimate_total[1],
                'M_total_COW.z (yaw)': self.M_total_above_water_in_centerline_csys[2] +
                                       self.M_centerline_csys_underwater_estimate_total[2],
                'M_total_COW.z (yaw - JG sign)': -(self.M_total_above_water_in_centerline_csys[2] +
                                                   self.M_centerline_csys_underwater_estimate_total[2]),
            }
            obj_as_dict.update(tmp)

        obj_as_dict.update(self.dyn_dict)
        df = pd.DataFrame.from_records(obj_as_dict, index=['Value']).transpose()
        df.reset_index(inplace=True)  # Convert the index (0-th like column) to 'regular' Column
        df = df.rename(columns={'index': 'Quantity'})
        return df
