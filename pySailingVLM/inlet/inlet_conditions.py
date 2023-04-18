import numpy as np
import pandas as pd
from pySailingVLM.inlet.winds import WindBase
from pySailingVLM.rotations.csys_transformations import CSYS_transformations

class InletConditions:
    def __init__(self, winds: WindBase, rho, center_of_pressure: np.ndarray):
        self.rho = rho  # fluid density [kg/m3]
        self.winds = winds
        self.cp_points = center_of_pressure
        self.tws_at_cp = np.array\
            ([self.winds.get_true_wind_speed_at_h(abs(cp_point[2])) for cp_point in self.cp_points])
        self.tws_length_at_ctr_points = np.linalg.norm(self.tws_at_cp, axis=1)

        self.V_app_infs = np.array\
            ([self.winds.get_app_infs_at_h(tws_at_ctr_point) for tws_at_ctr_point in self.tws_at_cp])
        self.V_app_infs_length = np.linalg.norm(self.V_app_infs, axis=1)

        self.AWA_infs_deg = np.rad2deg(np.arctan2(self.V_app_infs[:, 1], self.V_app_infs[:, 0]))

    def __iter__(self):
        yield 'cp_points_COG.x', self.cp_points[:, 0]
        yield 'cp_points_COG.y', self.cp_points[:, 1]
        yield 'cp_points_COG.z', self.cp_points[:, 2]
        yield 'TWS_COG.x', self.tws_at_cp[:, 0]
        yield 'TWS_COG.y', self.tws_at_cp[:, 1]
        yield 'TWS_COG.z', self.tws_at_cp[:, 2]
        yield 'TWS_length', self.tws_length_at_ctr_points
        yield 'V_app_infs_COG.x', self.V_app_infs[:, 0]
        yield 'V_app_infs_COG.y', self.V_app_infs[:, 1]
        yield 'V_app_infs_COG.z', self.V_app_infs[:, 2]
        yield 'V_app_infs_length', self.V_app_infs_length
        yield 'AWA_infs_COG_deg', self.AWA_infs_deg

    def to_df_full(self, csys_transformations: CSYS_transformations):
        obj_as_dict = dict(self)  # this calls __iter__

        tmp = {
            'AWA_infs_COW_deg': self.AWA_infs_deg - csys_transformations.leeway_deg,  # COW = centerline
        }
        obj_as_dict.update(tmp)
        df = pd.DataFrame.from_records(obj_as_dict)
        return df