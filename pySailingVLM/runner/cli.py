import numpy as np
import sys
import os
import argparse
import shutil

from pySailingVLM.rotations.csys_transformations import CSYS_transformations
from pySailingVLM.yacht_geometry.hull_geometry import HullGeometry
from pySailingVLM.results.save_utils import save_results_to_file
from pySailingVLM.solver.panels_plotter import display_panels_xyz_and_winds
from pySailingVLM.results.inviscid_flow import InviscidFlowResults
from pySailingVLM.solver.coefs import get_vlm_Cxyz
from pySailingVLM.solver.vlm import Vlm
from pySailingVLM.runner.sail import Wind, Sail

from pySailingVLM.solver.panels_plotter import plot_cp
def load_variable_module(args):
    try:
        sys.path.append(args.dvars)
        globals()['vr'] = __import__('variables')
    except ImportError as e:
        print(f'No vartiable.py file found in {args.dvars}\n{e}')
        sys.exit(0)
    
def parse_cli():

    parser = argparse.ArgumentParser(prog='pySailingVLM', description='Vortex lattice method for sailing')
    
    ##### required args
    requiredNamed = parser.add_argument_group('required arguments')
    #requiredNamed.add_argument('-v', '--vars', type=str, help="Directory containing input python file with variables, optional", default=os.getcwd(), required=False)   
    
    # dac z tego full path in relative

    # optional args
    # dir for storing output files
    #parser.add_argument('-d', '--dir', nargs='?', default=os.getcwd(), type=str, help="Alternative directory for for results, optional", required=False)
    parser.add_argument('-dv', '--dvars', type=str, help="Directory containing input python file with variables, optional", default=os.getcwd(), required=False)   
    # check if dv contain module!
    args = parser.parse_args()
    load_variable_module(args)
    return args                      


def main():
    parse_cli()
    csys_transformations = CSYS_transformations(
        vr.heel_deg, vr.leeway_deg,
        v_from_original_xyz_2_reference_csys_xyz=vr.reference_level_for_moments)

    w = Wind(vr.alpha_true_wind_deg, vr.tws_ref, vr.SOG_yacht, vr.wind_exp_coeff, vr.wind_reference_measurment_height, vr.reference_water_level_for_wind_profile, vr.roughness, vr.wind_profile)
    w_profile = w.profile

    s = Sail(vr.n_spanwise, vr.n_chordwise, csys_transformations, vr.sheer_above_waterline,
             vr.rake_deg, vr.boom_above_sheer, vr.mast_LOA,
             vr.main_sail_luff, vr.main_sail_girths, vr.main_sail_chords, vr.main_sail_centerline_twist_deg, vr.main_sail_camber, vr.main_sail_camber_distance_from_luff,
            vr.foretriangle_base, vr.foretriangle_height, 
            vr.jib_luff, vr.jib_girths, vr.jib_chords, vr.jib_centerline_twist_deg, vr.jib_sail_camber, vr.jib_sail_camber_distance_from_luff,
            vr.sails_def, vr.LLT_twist, vr.interpolation_type)
    sail_set = s.sail_set
    hull = HullGeometry(vr.sheer_above_waterline, vr.foretriangle_base, csys_transformations, vr.center_of_lateral_resistance_upright)
    myvlm = Vlm(sail_set.panels, vr.n_chordwise, vr.n_spanwise, vr.rho, w_profile, sail_set.trailing_edge_info, sail_set.leading_edge_info)

    inviscid_flow_results = InviscidFlowResults(sail_set, csys_transformations, myvlm)
    
    inviscid_flow_results.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)


    print("Preparing visualization.")   
    display_panels_xyz_and_winds(myvlm, inviscid_flow_results, myvlm.inlet_conditions, hull, show_plot=True)
    df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, csys_transformations, inviscid_flow_results, sail_set, vr.output_dir_name, vr.file_name)

    
    shutil.copy(os.path.join(vr.case_dir, vr.case_name), os.path.join(vr.output_dir_name, vr.case_name))

    print(f"-------------------------------------------------------------")
    print(f"Notice:\n"
          f"\tThe forces [N] and moments [Nm] are without profile drag.\n"
          f"\tThe the _COG_ CSYS is aligned in the direction of the yacht movement (course over ground).\n"
          f"\tThe the _COW_ CSYS is aligned along the centerline of the yacht (course over water).\n"
          f"\tNumber of panels (sail sail_set with mirror): {sail_set.panels.shape}")

    print(df_integrals)

    ##### 
    AR = 2 * vr.main_sail_luff / vr.main_sail_chords[0]
    S = 2*vr.main_sail_luff * vr.main_sail_chords[0]
    
    Cx_vlm, Cy_vlm, Cz_vlm= get_vlm_Cxyz(myvlm.force, np.array(w_profile.get_true_wind_speed_at_h(1.0)), vr.rho, S)
    print(f"C:[{Cx_vlm}, {Cy_vlm}, {Cz_vlm}]")#\nF_tot={tot_F}\nV={V} S={S} q={q}")

    plot_cp(sail_set.zero_mesh, myvlm.p_coeffs, vr.output_dir_name)
    