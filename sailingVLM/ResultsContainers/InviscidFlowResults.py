import numpy as np
import pandas as pd
from sailingVLM.Solver.forces import calc_moments, extract_above_water_quantities, calc_moment_arm_in_shifted_csys
from sailingVLM.Solver.forces import determine_vector_from_its_dot_and_cross_product
from sailingVLM.Rotations.CSYS_transformations import CSYS_transformations
from sailingVLM.YachtGeometry.SailGeometry import SailSet

from sailingVLM.Inlet.InletConditions import InletConditions
from sailingVLM.Solver.forces import calc_force_inviscid_xyz, calc_force_wrapper_new


def prepare_inviscid_flow_results(V_app_fs, V_induced, gamma_magnitude, v_ind_coeff,
                                  sail_set: SailSet,
                                  inletConditions: InletConditions,
                                  csys_transformations: CSYS_transformations):

    spans = np.array([p.get_span_vector() for p in sail_set.panels1d])
    # TODO: this shall be like in calc_force_wrapper:
    #  - V_app_fs with respect to cp
    #  - calculate dGamma as there are more panels in chordwise direction

    force_xyz = calc_force_inviscid_xyz(V_app_fs, gamma_magnitude, spans, inletConditions.rho)  # be carefull V_app_fs shall be calculated with respect to cp

    inviscid_flow_results = InviscidFlowResults(gamma_magnitude, v_ind_coeff, V_induced, V_app_fs,
                                                force_xyz, sail_set, csys_transformations)
    return inviscid_flow_results


class InviscidFlowResults:
    def __init__(self, gamma_magnitude, v_ind_coeff, V_induced, V_app_fs, force_xyz,
                 sail_set: SailSet,
                 csys_transformations: CSYS_transformations):

        cp_points = sail_set.get_cp_points()

        self.csys_transformations = csys_transformations
        self.gamma_magnitude = gamma_magnitude
        self.v_ind_coeff = v_ind_coeff
        self.V_induced = V_induced
        self.V_induced_length = np.linalg.norm(self.V_induced, axis=1)
        self.V_app_fs = V_app_fs
        self.V_app_fs_length = np.linalg.norm(self.V_app_fs, axis=1)
        self.AWA_app_fs = np.arctan(V_app_fs[:, 1] / V_app_fs[:, 0])
        # self.alfa_ind = alfa_app_infs - self.alfa_app_fs

        self.F_xyz = force_xyz
        F_xyz_above_water, self.F_xyz_total = extract_above_water_quantities(self.F_xyz, cp_points)

        # todo clean up the mess
        # todo check the output ... and add a test
        # F_xyz_above_water_jib = sail_set.extract_data_above_water_by_id(self.F_xyz, 0)
        # F_xyz_above_water_jib_total = np.sum(F_xyz_above_water_jib, axis=0)
        # xxx = sail_set.extract_data_above_water_to_df(force_xyz)

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

        self.M_xyz = calc_moments(r, force_xyz)
        _, self.M_total_above_water_in_xyz_csys = extract_above_water_quantities(self.M_xyz, cp_points)

        r_dot_F = np.array([np.dot(r[i], force_xyz[i]) for i in range(len(force_xyz))])
        _, r_dot_F_total_above_water = extract_above_water_quantities(r_dot_F, cp_points)

        self.above_water_centre_of_effort_estimate_xyz \
            = determine_vector_from_its_dot_and_cross_product(
                self.F_xyz_total, r_dot_F_total_above_water, self.M_total_above_water_in_xyz_csys)

        # r0 = self.M_total_above_water_in_xyz_csys[0] /self.F_xyz_total[0]
        # r1 = self.M_total_above_water_in_xyz_csys[1] / self.F_xyz_total[1]
        # r2 = self.M_total_above_water_in_xyz_csys[2] / self.F_xyz_total[2]
        # r_naive = np.array([r0,r1,r2])
        # np.cross(r_naive, self.F_xyz_total)

        self.F_centerline = csys_transformations.from_xyz_to_centerline_csys(force_xyz)
        _, self.F_centerline_total = extract_above_water_quantities(self.F_centerline, cp_points)

        self.M_centerline_csys = csys_transformations.from_xyz_to_centerline_csys(self.M_xyz)
        _, M_total_above_water_in_centerline_csys = extract_above_water_quantities(self.M_centerline_csys, cp_points)
        self.M_total_above_water_in_centerline_csys = M_total_above_water_in_centerline_csys

    def estimate_heeling_moment_from_keel(self, underwater_centre_of_effort_xyz):
        self.M_xyz_underwater_estimate_total = np.cross(underwater_centre_of_effort_xyz, -self.F_xyz_total)
        self.M_centerline_csys_underwater_estimate_total = self.csys_transformations.from_xyz_to_centerline_csys([self.M_xyz_underwater_estimate_total])[0]

    def __iter__(self):
        yield 'Circulation_magnitude', self.gamma_magnitude
        yield 'V_induced_COG.x', self.V_induced[:, 0]
        yield 'V_induced_COG.y', self.V_induced[:, 1]
        yield 'V_induced_COG.z', self.V_induced[:, 2]
        yield 'V_induced_length', self.V_induced_length
        yield 'V_app_fs_COG.x', self.V_app_fs[:, 0]
        yield 'V_app_fs_COG.y', self.V_app_fs[:, 1]
        yield 'V_app_fs_COG.z', self.V_app_fs[:, 2]
        yield 'V_app_fs_length', self.V_app_fs_length
        yield 'AWA_fs_COG_deg', np.rad2deg(self.AWA_app_fs)
        yield 'AWA_fs_COW_deg', np.rad2deg(self.AWA_app_fs) - self.csys_transformations.leeway_deg
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
                'M_total_COW.x (heel)': self.M_total_above_water_in_centerline_csys[0] + self.M_centerline_csys_underwater_estimate_total[0],
                'M_total_COW.y (pitch)': self.M_total_above_water_in_centerline_csys[1] + self.M_centerline_csys_underwater_estimate_total[1],
                'M_total_COW.z (yaw)': self.M_total_above_water_in_centerline_csys[2] + self.M_centerline_csys_underwater_estimate_total[2],
                'M_total_COW.z (yaw - JG sign)': -(self.M_total_above_water_in_centerline_csys[2] + self.M_centerline_csys_underwater_estimate_total[2]),
            }
            obj_as_dict.update(tmp)

        obj_as_dict.update(self.dyn_dict)
        df = pd.DataFrame.from_records(obj_as_dict, index=['Value']).transpose()
        df.reset_index(inplace=True)  # Convert the index (0-th like column) to 'regular' Column
        df = df.rename(columns={'index': 'Quantity'})
        return df
