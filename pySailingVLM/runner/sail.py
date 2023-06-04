from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pySailingVLM.runner.container import Rig, JibSail, MainSail, Solver, Conditions

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
    conditions: Conditions
    # alpha_true_wind_deg: float
    # tws_ref: float
    # SOG_yacht: float
    
    # wind_exp_coeff: float  = None # expotentional
    # wind_reference_measurment_height: float = None# expotentional or logarithmic
    # reference_water_level_for_wind_profile: float = None# expotentional
    # roughness: float = None # logarithmic
    # wind_profile: Literal['exponential', 'flat', 'logarithmic = 'exponential'
    
    def __post_init__(self):
        try: 
            if self.conditions.wind_profile == 'logarithmic' and None in [self.conditions.roughness, self.conditions.wind_reference_measurment_height]:
                raise ValueError("ERROR in logarithmic mode!: Roughness or wind_reference_measurment_height badly defined!")
            elif self.conditions.wind_profile == 'expotentional' and None in [self.conditions.wind_exp_coeff, self.conditions.wind_reference_measurment_height]:
                raise ValueError("ERROR in expotentional mode!: wind_exp_coeff or wind_reference_measurment_height badly defined!")
            elif self.conditions.wind_profile == 'flat' and None in [self.conditions.wind_exp_coeff, self.conditions.wind_reference_measurment_height]:
                raise ValueError("ERROR in flat mode!: wind_exp_coeff or wind_reference_measurment_height badly defined!")
        except ValueError as err:
            print(err)
            sys.exit()
            
    def __get_profile(self):
        if self.conditions.wind_profile == 'exponential':
            wind = ExpWindProfile(
                self.conditions.alpha_true_wind_deg, self.conditions.tws_ref, self.conditions.SOG_yacht,
                exp_coeff=self.conditions.wind_exp_coeff,
                reference_measurment_height=self.conditions.wind_reference_measurment_height,
                reference_water_level_for_wind_profile=self.conditions.reference_water_level_for_wind_profile)
        elif self.conditions.wind_profile == 'flat':
            wind = FlatWindProfile(self.conditions.alpha_true_wind_deg, self.conditions.tws_ref, self.conditions.SOG_yacht)
        else:
            wind = LogWindProfile(
                self.conditions.alpha_true_wind_deg, self.conditions.tws_ref, self.conditions.SOG_yacht,
                roughness=self.conditions.roughness,
                reference_measurment_height=self.conditions.wind_reference_measurment_height)
        return wind

    @property
    def profile(self):
        return self.__get_profile()
    
@dataclass
class Sail:
    
    solver: Solver
    rig: Rig
    main_sail: MainSail
    jib_sail: JibSail
    csys_transformations: CSYS_transformations
     
    def __post_init__(self):
        try:
            if self.jib_sail is not None:
                jib_stuff = [self.rig.foretriangle_base, self.rig.foretriangle_height, self.rig.jib_luff, self.jib_sail.girths, self.jib_sail.centerline_twist_deg, self.jib_sail.camber, self.jib_sail.camber_distance_from_luff]
            if self.main_sail is not None:
                main_stuff = [self.rig.main_sail_luff, self.main_sail.girths, self.main_sail.centerline_twist_deg, self.main_sail.camber, self.main_sail.camber_distance_from_luff]
            
            if self.rig.sails_def == 'jib' and check_list_for_none(jib_stuff): # if JIB and all vars are not NONE
                raise ValueError('Not all JIB variables are speciified!')
            elif self.rig.sails_def == 'main' and check_list_for_none(main_stuff):
                raise ValueError('Not all MAIN variables are speciified!') 
            elif self.rig.sails_def == 'jib_and_main' and check_list_for_none([*jib_stuff, *main_stuff]):
                raise ValueError('Not all JIB or MAIN variables are speciified!')
            
            if self.rig.sails_def == 'jib':
                check_input_arr_shapes(jib_stuff[3:], self.rig.sails_def)
            elif self.rig.sails_def =='main':
                check_input_arr_shapes(main_stuff[1:], self.rig.sails_def)
            else:
                check_input_arr_shapes(jib_stuff[3:], self.rig.sails_def)
                check_input_arr_shapes(main_stuff[1:], self.rig.sails_def)
        except ValueError as e:
            print(e)
            sys.exit()

    def __get_sail_set(self):
        
        interpolator = Interpolator(self.solver.interpolation_type)
        factory = SailFactory(csys_transformations=self.csys_transformations, n_spanwise=self.solver.n_spanwise, n_chordwise=self.solver.n_chordwise,
                                rake_deg=self.rig.rake_deg, sheer_above_waterline=self.rig.sheer_above_waterline)
        
        geoms = []
        if self.rig.sails_def == 'jib' or self.rig.sails_def == 'jib_and_main':
            jib_geometry = factory.make_jib(
                jib_luff=self.rig.jib_luff,
                foretriangle_base=self.rig.foretriangle_base,
                foretriangle_height=self.rig.foretriangle_height,
                jib_chords=interpolator.interpolate_girths(self.jib_sail.girths, self.jib_sail.chords, self.solver.n_spanwise + 1),
                sail_twist_deg=interpolator.interpolate_girths(self.jib_sail.girths, self.jib_sail.centerline_twist_deg, self.solver.n_spanwise + 1),
                mast_LOA=self.rig.mast_LOA,
                LLT_twist=self.solver.LLT_twist, 
                interpolated_camber=interpolator.interpolate_girths(self.jib_sail.girths, self.jib_sail.camber, self.solver.n_spanwise + 1),
                interpolated_distance_from_luff=interpolator.interpolate_girths(self.jib_sail.girths, self.jib_sail.camber_distance_from_luff, self.solver.n_spanwise + 1)
                )
            geoms.append(jib_geometry)
            
        if self.rig.sails_def == 'main' or self.rig.sails_def =='jib_and_main':
            main_sail_geometry = factory.make_main_sail(
                main_sail_luff=self.rig.main_sail_luff,
                boom_above_sheer=self.rig.boom_above_sheer,
                main_sail_chords=interpolator.interpolate_girths(self.main_sail.girths, self.main_sail.chords, self.solver.n_spanwise + 1),
                sail_twist_deg=interpolator.interpolate_girths(self.main_sail.girths, self.main_sail.centerline_twist_deg, self.solver.n_spanwise + 1),
                LLT_twist=self.solver.LLT_twist,
                interpolated_camber=interpolator.interpolate_girths(self.main_sail.girths, self.main_sail.camber, self.solver.n_spanwise + 1),
                interpolated_distance_from_luff=interpolator.interpolate_girths(self.main_sail.girths, self.main_sail.camber_distance_from_luff, self.solver.n_spanwise + 1)
                )
            geoms.append(main_sail_geometry)

        return SailSet(geoms)
    
    @property
    def sail_set(self):
        return self.__get_sail_set()