import sys
import numpy as np
from dataclasses import dataclass

from typing import Literal, Optional
from pySailingVLM.inlet.winds import ExpWindProfile, FlatWindProfile, LogWindProfile
from pySailingVLM.rotations.csys_transformations import CSYS_transformations
from pySailingVLM.yacht_geometry.sail_factory import SailFactory
from pySailingVLM.yacht_geometry.sail_geometry import SailSet
from pySailingVLM.solver.interpolator import Interpolator

def check_list_for_none(list_of_different_obj: list):
    # list like: a = [np.array([1.9]), 1.0, None] should return true
    # a = [np.array([1.9]), 1.0] should return false
    res = False
    for element in list_of_different_obj:
        if element is None:
            res = True
            break
    return res

def check_input_arr_shapes(check_arr : list, sails_def: str):
    for arr in check_arr:
        if arr.shape != check_arr[0].shape:
            raise ValueError(f'Not all {sails_def.upper()} arrays are equal size!')    
@dataclass
class Wind:
    ## czy dataclass sprawdza type of input ???
    conditions: dict
    # alpha_true_wind_deg: float
    # tws_ref: float
    # SOG_yacht: float
    
    # wind_exp_coeff: float  = None # expotentional
    # wind_reference_measurment_height: float = None# expotentional or logarithmic
    # reference_water_level_for_wind_profile: float = None# expotentional
    # roughness: float = None # logarithmic
    # wind_profile: Literal['exponential', 'flat', 'logarithmic'] = 'exponential'
    
    def __post_init__(self):
        try: 
            # check if all these params are inside dictionary
            params = ['alpha_true_wind_deg', 'tws_ref', 'SOG_yacht', 'wind_exp_coeff', 'wind_reference_measurment_height', 'reference_water_level_for_wind_profile', 'roughness', 'wind_profile' ]
            if not all(k in self.conditions for k in (params)):
                raise ValueError(f"Not all keys are passes to conditions dict.\nIt should contain these keys {params}")
                
            if self.conditions['wind_profile'] == 'logarithmic' and None in [self.conditions['roughness'], self.conditions['wind_reference_measurment_height']]:
                raise ValueError("ERROR in logarithmic mode!: Roughness or wind_reference_measurment_height badly defined!")
            elif self.conditions['wind_profile'] == 'expotentional' and None in [self.conditions['wind_exp_coeff'], self.conditions['wind_reference_measurment_height']]:
                raise ValueError("ERROR in expotentional mode!: wind_exp_coeff or wind_reference_measurment_height badly defined!")
            elif self.conditions['wind_profile'] == 'flat' and None in [self.conditions['wind_exp_coeff'], self.conditions['wind_reference_measurment_height']]:
                raise ValueError("ERROR in flat mode!: wind_exp_coeff or wind_reference_measurment_height badly defined!")
        except ValueError as err:
            print(err)
            sys.exit()
            
    def __get_profile(self):
        if self.conditions['wind_profile'] == 'exponential':
            wind = ExpWindProfile(
                self.conditions['alpha_true_wind_deg'], self.conditions['tws_ref'], self.conditions['SOG_yacht'],
                exp_coeff=self.conditions['wind_exp_coeff'],
                reference_measurment_height=self.conditions['wind_reference_measurment_height'],
                reference_water_level_for_wind_profile=self.conditions['reference_water_level_for_wind_profile'])
        elif self.conditions['wind_profile'] == 'flat':
            wind = FlatWindProfile(self.conditions['alpha_true_wind_deg'], self.conditions['tws_ref'], self.conditions['SOG_yacht'])
        else:
            wind = LogWindProfile(
                self.conditions['alpha_true_wind_deg'], self.conditions['tws_ref'], self.conditions['SOG_yacht'],
                roughness=self.conditions['roughness'],
                reference_measurment_height=self.conditions['wind_reference_measurment_height'])
        return wind

    @property
    def profile(self):
        return self.__get_profile()
    
@dataclass
class Sail:
    
    solver: dict
    rig: dict
    main_sail: dict
    jib_sail: dict
    csys_transformations: CSYS_transformations
     
    def __post_init__(self):

        solver_params = ['n_spanwise',  'n_chordwise','interpolation_type', 'LLT_twist']
        rig_params = ['main_sail_luff','jib_luff','foretriangle_height','foretriangle_base','sheer_above_waterline','boom_above_sheer','rake_deg','mast_LOA','sails_def','main_sail_girths','jib_girths']
        main_sail_params = ['main_sail_chords','main_sail_centerline_twist_deg','main_sail_camber','main_sail_camber_distance_from_luff']
        jib_sail_params = ['jib_sail_camber','jib_sail_camber_distance_from_luff','jib_chords','jib_centerline_twist_deg']
        
        try:
            if not all(k in self.solver for k in (solver_params)):
                raise ValueError(f"Not all keys are passes to solver dict.\nIt should contain these keys {solver_params}")
            
            if not all(k in self.rig for k in (rig_params)):
                raise ValueError(f"Not all keys are passes to rig dict.\nIt should contain these keys {rig_params}")
            
            if not all(k in self.main_sail for k in (main_sail_params)):
                raise ValueError(f"Not all keys are passes to rig dict.\nIt should contain these keys {main_sail_params}")
           
            if not all(k in self.jib_sail for k in (jib_sail_params)):
                raise ValueError(f"Not all keys are passes to rig dict.\nIt should contain these keys {jib_sail_params}")

            jib_stuff = [self.rig['foretriangle_base'], self.rig['foretriangle_height'], self.rig['jib_luff'], self.rig['jib_girths'], self.rig['jib_girths'], self.jib_sail['jib_centerline_twist_deg'], self.jib_sail['jib_sail_camber'], self.jib_sail['jib_sail_camber_distance_from_luff']]
            
            main_stuff = [self.rig['main_sail_luff'], self.rig['main_sail_girths'], self.main_sail['main_sail_centerline_twist_deg'], self.main_sail['main_sail_camber'], self.main_sail['main_sail_camber_distance_from_luff']]
            if self.rig['sails_def'] == 'jib' and check_list_for_none(jib_stuff): # if JIB and all vars are not NONE
                raise ValueError('Not all JIB variables are speciified!')
            elif self.rig['sails_def'] == 'main' and check_list_for_none(main_stuff):
                raise ValueError('Not all MAIN variables are speciified!') 
            elif self.rig['sails_def'] == 'jib_and_main' and check_list_for_none([*jib_stuff, *main_stuff]):
                raise ValueError('Not all JIB or MAIN variables are speciified!')
            
            if self.rig['sails_def'] == 'jib':
                check_input_arr_shapes(jib_stuff[3:], self.rig['sails_def'])
            elif self.rig['sails_def'] =='main':
                check_input_arr_shapes(main_stuff[1:], self.rig['sails_def'])
            else:
                check_input_arr_shapes(jib_stuff[3:], self.rig['sails_def'])
                check_input_arr_shapes(main_stuff[1:], self.rig['sails_def'])
        except ValueError as e:
            print(e)
            sys.exit()

    def __get_sail_set(self):
        
        interpolator = Interpolator(self.solver['interpolation_type'])
        factory = SailFactory(csys_transformations=self.csys_transformations, n_spanwise=self.solver['n_spanwise'], n_chordwise=self.solver['n_chordwise'],
                                rake_deg=self.rig['rake_deg'], sheer_above_waterline=self.rig['sheer_above_waterline'])
        
        geoms = []
        if self.rig['sails_def'] == 'jib' or self.rig['sails_def'] == 'jib_and_main':
            jib_geometry = factory.make_jib(
                jib_luff=self.rig['jib_luff'],
                foretriangle_base=self.rig['foretriangle_base'],
                foretriangle_height=self.rig['foretriangle_height'],
                jib_chords=interpolator.interpolate_girths(self.rig['jib_girths'], self.jib_sail['jib_chords'], self.solver['n_spanwise'] + 1),
                sail_twist_deg=interpolator.interpolate_girths(self.rig['jib_girths'], self.jib_sail['jib_centerline_twist_deg'], self.solver['n_spanwise'] + 1),
                mast_LOA=self.rig['mast_LOA'],
                LLT_twist=self.solver['LLT_twist'], 
                interpolated_camber=interpolator.interpolate_girths(self.rig['jib_girths'], self.jib_sail['jib_sail_camber'], self.solver['n_spanwise'] + 1),
                interpolated_distance_from_luff=interpolator.interpolate_girths(self.rig['jib_girths'], self.jib_sail['jib_sail_camber_distance_from_luff'], self.solver['n_spanwise'] + 1)
                )
            geoms.append(jib_geometry)
            
        if self.rig['sails_def'] == 'main' or self.rig['sails_def'] =='jib_and_main':
            main_sail_geometry = factory.make_main_sail(
                main_sail_luff=self.rig['main_sail_luff'],
                boom_above_sheer=self.rig['boom_above_sheer'],
                main_sail_chords=interpolator.interpolate_girths(self.rig['main_sail_girths'], self.main_sail['main_sail_chords'], self.solver['n_spanwise'] + 1),
                sail_twist_deg=interpolator.interpolate_girths(self.rig['main_sail_girths'], self.main_sail['main_sail_centerline_twist_deg'], self.solver['n_spanwise'] + 1),
                LLT_twist=self.solver['LLT_twist'],
                interpolated_camber=interpolator.interpolate_girths(self.rig['main_sail_girths'], self.main_sail['main_sail_camber'], self.solver['n_spanwise'] + 1),
                interpolated_distance_from_luff=interpolator.interpolate_girths(self.rig['main_sail_girths'], self.main_sail['main_sail_camber_distance_from_luff'], self.solver['n_spanwise'] + 1)
                )
            geoms.append(main_sail_geometry)

        return SailSet(geoms)
    
    @property
    def sail_set(self):
        return self.__get_sail_set()