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
from pySailingVLM.solver.vlm import Vlm
from pySailingVLM.runner.sail import Wind, Sail
from pySailingVLM.solver.panels_plotter import plot_cp, plot_section_coeff
from pySailingVLM.runner.container import Output, Rig, Conditions, Solver, MainSail, JibSail, Csys, Keel
from pySailingVLM.solver.coefs import get_data_for_coeff_plot

def load_variable_module(args):
    try:
        sys.path.append(args.dvars)
        globals()['vr'] = __import__('variables')
    except ImportError as e:
        print(f'No variables.py file found in {args.dvars}\n{e}')
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
    #start = timeit.default_timer()
    parse_cli()
    out = Output(**vr.output_args)
    conditions = Conditions(**vr.conditions_args)
    solver = Solver(**vr.solver_args)
    main = MainSail(**vr.main_sail_args)
    jib = JibSail(**vr.jib_sail_args)
    csys = Csys(**vr.csys_args)
    keel = Keel(**vr.keel_args)
    rig = Rig(**vr.rig_args)
    
    csys_transformations = CSYS_transformations(
        conditions.heel_deg, conditions.leeway_deg,
        v_from_original_xyz_2_reference_csys_xyz=csys.reference_level_for_moments)

    
    w = Wind(conditions)
    s = Sail(solver, rig, main, jib, csys_transformations)
    sail_set = s.sail_set
    hull = HullGeometry(rig.sheer_above_waterline, rig.foretriangle_base, csys_transformations, keel.center_of_lateral_resistance_upright)
    myvlm = Vlm(sail_set.panels, solver.n_chordwise, solver.n_spanwise, conditions.rho, w.profile, sail_set.trailing_edge_info, sail_set.leading_edge_info)

    inviscid_flow_results = InviscidFlowResults(sail_set, csys_transformations, myvlm)
    inviscid_flow_results.estimate_heeling_moment_from_keel(hull.center_of_lateral_resistance)

    
    print(f"-------------------------------------------------------------")
    print(f"Notice:\n"
          f"\tThe forces [N] and moments [Nm] are without profile drag.\n"
          f"\tThe the _COG_ CSYS is aligned in the direction of the yacht movement (course over ground).\n"
          f"\tThe the _COW_ CSYS is aligned along the centerline of the yacht (course over water).\n"
          f"\tNumber of panels (sail sail_set with mirror): {sail_set.panels.shape}")

    df_components, df_integrals, df_inlet_IC = save_results_to_file(myvlm, csys_transformations, inviscid_flow_results, sail_set, out.name, out.file_name)
    shutil.copy(os.path.join(out.case_dir, out.case_name), os.path.join(out.name, out.case_name))

    print(df_integrals)
    #print(f"\nCPU time: {float(timeit.default_timer() - start):.2f} [s]")
    
    print("Preparing visualization.")   
    display_panels_xyz_and_winds(myvlm, inviscid_flow_results, myvlm.inlet_conditions, hull, show_plot=True, show_induced_wind=False, is_sailopt_mode=False)
    
    plot_cp(sail_set.zero_mesh, myvlm.p_coeffs, out.name)
    
    mean_cp, cl_data, cd_data = get_data_for_coeff_plot(myvlm, solver)
    plot_section_coeff(cd_data, mean_cp,  out.name,  'lift', ['blue', 'green'])
    plot_section_coeff(cl_data, mean_cp, out.name,  'drag',  ['teal', 'purple'])
    