import sys
import numpy as np
from dataclasses import dataclass
from varname import varname
from typing import Literal, Optional
from sailing_vlm.inlet.winds import ExpWindProfile, FlatWindProfile, LogWindProfile
from sailing_vlm.rotations.csys_transformations import CSYS_transformations
from sailing_vlm.yacht_geometry.sail_factory import SailFactory
from sailing_vlm.yacht_geometry.sail_geometry import SailSet
from sailing_vlm.solver.interpolator import Interpolator


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

@dataclass
class Sail:
    
    sails_def: Literal['jib', 'main', 'jib_and_main'] = 'main'
    n_spanwise: int
    n_chordwise: int
    LLT_twist: Literal['real_twist', 'average_const', 'sheeting_angle_const'] = 'main'
    interpolation_type: Literal['spline', 'linear'] = 'spline'
    csys_transformations: CSYS_transformations
    
    sheer_above_waterline: float
    rake_deg: float
    boom_above_sheer: float
    mast_LOA: float
    
    
    foretriangle_base: Optional[float] = None
    foretriangle_height: Optional[float] = None
    jib_luff: Optional[float] = None
    jib_girths:  Optional[np.array] = None
    jib_chords: Optional[np.array] = None
    jib_centerline_twist_deg: Optional[np.array] = None
    jib_sail_camber: Optional[np.array] = None
    jib_sail_camber_distance_from_luff: Optional[np.array] = None
    
    main_sail_luff: float
    main_sail_girths: np.array
    main_sail_chords: np.array
    main_sail_centerline_twist_deg: np.array
    main_sail_camber: np.array
    main_sail_camber_distance_from_luff: np.array
    
    def __post_init__(self):
        jib_stuff = [self.foretriangle_base, self.foretriangle_height, self.self.jib_luff, self.jib_girths, self.jib_girths, self.jib_centerline_twist_deg, self.jib_sail_camber, self.jib_sail_camber_distance_from_luff]
        main_stuff = [self.self.main_sail_luff, self.main_sail_girths, self.main_sail_girths, self.main_sail_centerline_twist_deg, self.main_sail_camber, self.main_sail_camber_distance_from_luff]
        try:
            if self.sails_def == 'jib' and not all(jib_stuff): # if JIB and all vars are not NONE
                raise ValueError('Not all JIB variables are speciified!')
            elif self.sails_def == 'main' and not all(main_stuff):
                raise ValueError('Not all MAIN variables are speciified!') 
            elif self.sails_def == 'jib_and_main' and not (all(jib_stuff) and all(main_stuff)):
                raise ValueError('Not all JIB or MAIN variables are speciified!')
            
            if self.sails_def == 'jib':
                check_arr = jib_stuff[3:]
            elif self.sails_def =='main':
                check_arr = main_stuff[1:]
            else:
                check_arr = jib_stuff[3:] + main_stuff[1:]
                
            if not np.all(check_arr == check_arr[0]):
                raise ValueError(f'Not all {self.sails_def.upper()} arrays are equal size!')
                
        except ValueError as e:
            print(e.err)
            sys.exit()

    def __get_sail_set(self):
        
        interpolator = Interpolator(self.interpolation_type)
        factory = SailFactory(csys_transformations=self.csys_transformations, n_spanwise=self.n_spanwise, n_chordwise=self.n_chordwise,
                                rake_deg=self.rake_deg, sheer_above_waterline=self.sheer_above_waterline)
        
        geoms = []
        if self.sails_def == 'jib' or self.sails_def == 'jib_and_main':
            jib_geometry = factory.make_jib(
                jib_luff=self.jib_luff,
                foretriangle_base=self.foretriangle_base,
                foretriangle_height=self.foretriangle_height,
                jib_chords=interpolator.interpolate_girths(self.jib_girths, self.jib_chords, self.n_spanwise + 1),
                sail_twist_deg=interpolator.interpolate_girths(self.jib_girths, self.jib_centerline_twist_deg, self.n_spanwise + 1),
                mast_LOA=self.mast_LOA,
                LLT_twist=self.LLT_twist, 
                interpolated_camber=interpolator.interpolate_girths(self.jib_girths, self.jib_sail_camber, self.n_spanwise + 1),
                interpolated_distance_from_luff=interpolator.interpolate_girths(self.jib_girths, self.jib_sail_camber_distance_from_luff, self.n_spanwise + 1)
                )
            geoms.append(jib_geometry)
            
        if self.sails_def == 'main' or self.sails_def =='jib_and_main':
            main_sail_geometry = factory.make_main_sail(
                main_sail_luff=self.main_sail_luff,
                boom_above_sheer=self.boom_above_sheer,
                main_sail_chords=interpolator.interpolate_girths(self.main_sail_girths, self.main_sail_chords, self.n_spanwise + 1),
                sail_twist_deg=interpolator.interpolate_girths(self.main_sail_girths, self.main_sail_centerline_twist_deg, n_spanwise + 1),
                LLT_twist=self.LLT_twist,
                interpolated_camber=interpolator.interpolate_girths(self.main_sail_girths, self.main_sail_camber, self.n_spanwise + 1),
                interpolated_distance_from_luff=interpolator.interpolate_girths(self.main_sail_girths, self.main_sail_camber_distance_from_luff, self.n_spanwise + 1)
                )
            geoms.append(main_sail_geometry)

        return SailSet(geoms)
    
    @property
    def sail_set(self):
        return self.__get_sail_set()