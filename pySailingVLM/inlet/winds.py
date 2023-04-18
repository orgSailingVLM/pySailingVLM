import numpy as np
import pandas as pd
from abc import abstractmethod
from pySailingVLM.rotations.csys_transformations import CSYS_transformations


class WindBase:
    def __init__(self, alpha_true_wind_deg, tws_ref, SOG_yacht):
        """
        Parameters
        ----------
        alpha_true_wind_deg - [deg] angle between true wind and direction of boat movement (including leeway)
        tws -  Free stream of true wind having velocity [m/s] at height z = 10 [m]
        """

        self.alpha_true_wind = np.deg2rad(alpha_true_wind_deg)  # [rad] angle between true wind and direction of boat movement (including leeway)
        self.tws_ref = tws_ref
        self.V_yacht = SOG_yacht

    @abstractmethod
    def get_true_wind_speed_at_h(self, height):
        # calc_tws_at_h = lambda h, tws_ref: tws_ref * (np.log((h) / self.roughness) / np.log(10 / self.roughness))
        # wind_speed = np.array([calc_tws_at_h(h, tws_ref) for h in heights])
        pass

    @abstractmethod
    def to_df_integral(self, csys_transformations: CSYS_transformations):
        pass

    def length_to_xyz_vector(self, wind_speed):
        tws = np.array([wind_speed * np.cos(self.alpha_true_wind),
                        wind_speed * np.sin(self.alpha_true_wind),
                        0])

        return tws

    def get_app_alfa_infs_at_h(self, tws_at_h):
        # alfa_yacht - angle between apparent wind and true wind
        # alfa_app_infs	- angle between apparent wind and direction of boat movement (including leeway)
        # (model of an 'infinite sail' is assumed == without induced wind velocity) and direction of boat movement (including leeway)

        tws_l = np.linalg.norm(tws_at_h)  # true wind speed - length of the vector
        # tws_l = np.sqrt(tws_at_h[0]*tws_at_h[0]+tws_at_h[1]*tws_at_h[1])  # true wind speed

        alfa_yacht = np.arccos((self.V_yacht * np.cos(self.alpha_true_wind) + tws_l) /
                               np.sqrt(
                                   self.V_yacht*self.V_yacht
                                   + 2*self.V_yacht*tws_l*np.cos(self.alpha_true_wind)
                                   + tws_l*tws_l))  # result in [rad]

        alpha_app_infs = self.alpha_true_wind - alfa_yacht  # result in [rad]

        return alpha_app_infs

    def get_app_infs_at_h(self, tws_at_h):
        aw_infs = np.array([tws_at_h[0] + self.V_yacht, tws_at_h[1], tws_at_h[2]])
        return aw_infs


class FlatWindProfile(WindBase):
    def __init__(self, alpha_true_wind_deg, tws_ref, SOG_yacht):
        super().__init__(alpha_true_wind_deg, tws_ref, SOG_yacht)

    def get_true_wind_speed_at_h(self, height):
        tws_at_h = self.length_to_xyz_vector(self.tws_ref)
        return tws_at_h

    def to_df_integral(self, csys_transformations: CSYS_transformations):
        obj_as_dict = {
            'Not implemented - dummy variable': 123456789,
        }
        df = pd.DataFrame.from_records(obj_as_dict, index=['Value']).transpose()
        df.reset_index(inplace=True)  # Convert the index (0-th like column) to 'regular' Column
        df = df.rename(columns={'index': 'Quantity'})
        return df


class LogWindProfile(WindBase):
    """
    roughness - Over smooth, open water, expect a value around 0.0002 m,
    while over flat, open grassland  ~ 0.03 m,
    cropland ~ 0.1-0.25 m, and brush or forest ~ 0.5-1.0 m
    """
    def __init__(self, alpha_true_wind_deg, tws_ref, SOG_yacht, roughness=0.05, reference_measurment_height=10.):
        super().__init__(alpha_true_wind_deg, tws_ref, SOG_yacht)
        self.roughness = roughness
        self.reference_measurment_height = reference_measurment_height

    def get_true_wind_speed_at_h(self, height):
        #  tws -  Free stream of true wind having velocity [m/s] at height z = 10 [m]
        wind_speed = self.tws_ref * (np.log((height) / self.roughness) / np.log(self.reference_measurment_height / self.roughness))
        tws_at_h = self.length_to_xyz_vector(wind_speed)
        return tws_at_h

    def to_df_integral(self, csys_transformations: CSYS_transformations):
        obj_as_dict = {
            'Not implemented - dummy variable': 123456789,
        }
        df = pd.DataFrame.from_records(obj_as_dict, index=['Value']).transpose()
        df.reset_index(inplace=True)  # Convert the index (0-th like column) to 'regular' Column
        df = df.rename(columns={'index': 'Quantity'})
        return df

class ExpWindProfile(WindBase):
    def __init__(self, alpha_true_wind_deg, tws_ref, SOG_yacht,
                 exp_coeff=0.1428,
                 reference_measurment_height=10.,
                 reference_water_level_for_wind_profile=0.):
        super().__init__(alpha_true_wind_deg, tws_ref, SOG_yacht)
        self.exp_coeff = exp_coeff
        self.reference_measurment_height = reference_measurment_height
        self.reference_water_level_for_wind_profile = reference_water_level_for_wind_profile
        # reference_waterline_level_for_wind_profile - this is an attempt to mimick the deck effect by lowering the sheer_above_waterline (sails' mirror)
        # while keeping the wind profile as in original geometry

    def get_true_wind_speed_at_h(self, height):
        wind_speed = self.tws_ref*pow((height - self.reference_water_level_for_wind_profile) / self.reference_measurment_height, self.exp_coeff)
        tws_at_h = self.length_to_xyz_vector(wind_speed)
        return tws_at_h

    def to_df_integral(self, csys_transformations: CSYS_transformations):
        tws_at_reference_height = self.get_true_wind_speed_at_h(self.reference_measurment_height)
        tws_length_at_reference_height = np.linalg.norm(tws_at_reference_height)

        V_app_infs_at_reference_height = self.get_app_infs_at_h(tws_at_reference_height)
        V_app_infs_length_at_reference_height = np.linalg.norm(V_app_infs_at_reference_height)
        AWA_infs_at_reference_height_deg = np.rad2deg(
            np.arctan2(V_app_infs_at_reference_height[1], V_app_infs_at_reference_height[0]))

        obj_as_dict = {
            'Reference_measurement_height': self.reference_measurment_height,
            'Reference_water_level_for_wind_profile': self.reference_water_level_for_wind_profile,
            'TWS_at_reference_height_COG.x': tws_at_reference_height[0],
            'TWS_at_reference_height_COG.y': tws_at_reference_height[1],
            'TWS_at_reference_height_COG.z': tws_at_reference_height[2],
            'TWS_length_at_reference_height': tws_length_at_reference_height,
            'V_app_infs_at_reference_height_COG.x': V_app_infs_at_reference_height[0],
            'V_app_infs_at_reference_height_COG.y': V_app_infs_at_reference_height[1],
            'V_app_infs_at_reference_height_COG.z': V_app_infs_at_reference_height[2],
            'V_app_infs_length_at_reference_height_COG': V_app_infs_length_at_reference_height,
            'AWA_infs_at_reference_height_COG_deg': AWA_infs_at_reference_height_deg,
            'AWA_infs_at_reference_height_COW_deg': AWA_infs_at_reference_height_deg - csys_transformations.leeway_deg,
        }
        df = pd.DataFrame.from_records(obj_as_dict, index=['Value']).transpose()
        df.reset_index(inplace=True)  # Convert the index (0-th like column) to 'regular' Column
        df = df.rename(columns={'index': 'Quantity'})
        return df
