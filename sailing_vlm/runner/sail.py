import sys
from dataclasses import dataclass
from typing import Literal
from sailing_vlm.inlet.winds import ExpWindProfile, FlatWindProfile, LogWindProfile

@dataclass
class Wind:
    ## czy dataclass sprawdza type of input ???
    wind_profile: Literal['exponential', 'flat', 'logarithmic'] = 'exponential'
    alpha_true_wind_deg: float
    tws_ref: float
    SOG_yacht: float
    
    wind_exp_coeff: float  = None # expotentional
    wind_reference_measurment_height: float = None# expotentional or logarithmic
    reference_water_level_for_wind_profile: float = None# expotentional
    roughness: float = None # logarithmic
    
    def __post__init__(self):
        # stestowac to
        try:
            if (self.wind_profile == 'logarithmic' and isinstance(self.roughness, float) and isinstance(self.wind_reference_measurment_height, float)):
                raise ValueError("ERROR in logarithmic mode!: Roughness or wind_reference_measurment_height badly defined!")
            elif (self.wind_profile == 'expotentional' and isinstance(self.wind_exp_coeff, float) and isinstance(self.wind_reference_measurment_height, float)):
                raise ValueError("ERROR in expotentional mode!: wind_exp_coeff or wind_reference_measurment_height badly defined!")
            elif (self.wind_profile == 'flat' and isinstance(self.wind_exp_coeff, float) and isinstance(self.wind_reference_measurment_height, float)):
                raise ValueError("ERROR in expotentional mode!: wind_exp_coeff or wind_reference_measurment_height badly defined!")
        except ValueError as err:
            print(err)
            sys.exit()
            
    def get_profile(self):
        if self.wind_profile == 'exponential':
            wind = ExpWindProfile(
                self.alpha_true_wind_deg, self.tws_ref, self.SOG_yacht,
                exp_coeff=self.wind_exp_coeff,
                reference_measurment_height=self.wind_reference_measurment_height,
                reference_water_level_for_wind_profile=self.reference_water_level_for_wind_profile)
        elif self.wind_profile == 'flat':
            wind = FlatWindProfile(self.alpha_true_wind_deg, self.tws_ref, self.SOG_yacht)
        else:
            wind = LogWindProfile(
                self.alpha_true_wind_deg, self.tws_ref, self.SOG_yacht,
                roughness=self.roughness,
                reference_measurment_height=self.wind_reference_measurment_height)
        return wind