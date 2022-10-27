import numpy as np
import pandas as pd

from sailingVLM.Solver.forces import calc_moments, calc_moment_arm_in_shifted_csys
from sailingVLM.Solver.forces import determine_vector_from_its_dot_and_cross_product
from sailingVLM.Rotations.CSYS_transformations import CSYS_transformations
from sailingVLM.YachtGeometry.SailGeometry import SailSet



from sailingVLM.NewApproach.vlm_logic import extract_above_water_quantities_new_approach

from sailingVLM.NewApproach.vlm import NewVlm


def prepare_inviscid_flow_results_vlm_new_approach(
                                      sail_set: SailSet,
                                      csys_transformations: CSYS_transformations, myvlm : NewVlm):

    inviscid_flow_results_new_approach = InviscidFlowResultsNew(sail_set, csys_transformations, myvlm)
    
    return inviscid_flow_results_new_approach


class InviscidFlowResultsNew:
    def __init__(self,
                 sail_set: SailSet,
                 csys_transformations: CSYS_transformations, myvlm : NewVlm):


        self.csys_transformations = csys_transformations
        self.gamma_magnitude = myvlm.gamma_magnitude
        self.pressure = myvlm.pressure
        self.V_induced_at_cp = myvlm.V_induced_at_cp#V_induced
        self.V_induced_length = np.linalg.norm(self.V_induced_at_cp, axis=1)
        self.V_app_fs_at_cp = myvlm.V_app_fs_at_cp#V_app_fs
        self.V_app_fs_length = np.linalg.norm(self.V_app_fs_at_cp, axis=1)
        self.AWA_app_fs = np.arctan(myvlm.V_app_fs_at_cp[:, 1] / myvlm.V_app_fs_at_cp[:, 0])
  
        self.F_xyz = myvlm.force
        self.F_xyz_above_water, self.F_xyz_total = extract_above_water_quantities_new_approach(self.F_xyz, myvlm.center_of_pressure)

        r = calc_moment_arm_in_shifted_csys(myvlm.center_of_pressure, csys_transformations.v_from_original_xyz_2_reference_csys_xyz)
     
        dyn_dict = {}
        # for jib and main, quantities is always dividable by 2
        half = int(myvlm.force.shape[0] / 2)
        # dla jiba i maina mamy: jib above, main_above
        # self.F_xyz[0:half] - gorna polowa paneli czyli ta nad woda
        # forces_chunks_above - gorne panele podzielone na chanki, 0 - jib, 1 - main
        forces_chunks_above = np.array_split(self.F_xyz[0:half], len(sail_set.sails))
        r_chunks_above = np.array_split(r[0:half], len(sail_set.sails))

        for i in range(len(sail_set.sails)):

            F_xyz_above_water_tmp = forces_chunks_above[i]
            r_tmp = r_chunks_above[i]

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

        self.M_xyz = calc_moments(r, myvlm.force)
        _, self.M_total_above_water_in_xyz_csys = extract_above_water_quantities_new_approach(self.M_xyz, myvlm.center_of_pressure)

        r_dot_F = np.array([np.dot(r[i], myvlm.force[i]) for i in range(len(myvlm.force))])
        _, r_dot_F_total_above_water = extract_above_water_quantities_new_approach(r_dot_F, myvlm.center_of_pressure)

        self.above_water_centre_of_effort_estimate_xyz \
            = determine_vector_from_its_dot_and_cross_product(
                self.F_xyz_total, r_dot_F_total_above_water, self.M_total_above_water_in_xyz_csys)

        # r0 = self.M_total_above_water_in_xyz_csys[0] /self.F_xyz_total[0]
        # r1 = self.M_total_above_water_in_xyz_csys[1] / self.F_xyz_total[1]
        # r2 = self.M_total_above_water_in_xyz_csys[2] / self.F_xyz_total[2]
        # r_naive = np.array([r0,r1,r2])
        # np.cross(r_naive, self.F_xyz_total)

        self.F_centerline = csys_transformations.from_xyz_to_centerline_csys(myvlm.force)
        _, self.F_centerline_total = extract_above_water_quantities_new_approach(self.F_centerline, myvlm.center_of_pressure)

        self.M_centerline_csys = csys_transformations.from_xyz_to_centerline_csys(self.M_xyz)
        _, M_total_above_water_in_centerline_csys = extract_above_water_quantities_new_approach(self.M_centerline_csys, myvlm.center_of_pressure)
        self.M_total_above_water_in_centerline_csys = M_total_above_water_in_centerline_csys

    def estimate_heeling_moment_from_keel(self, underwater_centre_of_effort_xyz):
        self.M_xyz_underwater_estimate_total = np.cross(underwater_centre_of_effort_xyz, -self.F_xyz_total)
        self.M_centerline_csys_underwater_estimate_total = self.csys_transformations.from_xyz_to_centerline_csys([self.M_xyz_underwater_estimate_total])[0]

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
