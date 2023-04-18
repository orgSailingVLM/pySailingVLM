import pandas as pd
import os
import numpy as np
from pySailingVLM.results.inviscid_flow import InviscidFlowResults

from typing import List
from pySailingVLM.inlet.inlet_conditions import InletConditions
from pySailingVLM.yacht_geometry.sail_geometry import SailSet

from pySailingVLM.solver.additional_functions import get_cp_z_as_girths_all, get_cp_z_as_girths_all_above

def save_results_to_file(myvlm, csys_transformations,
                         inviscid_flow_results: InviscidFlowResults,
                         sail_set: SailSet,
                         output_dir="output", file_name='results'):
    cp_z_as_girths_all, cp_straight_yacht_all = get_cp_z_as_girths_all(sail_set, csys_transformations, myvlm.cp)
    cp_z_as_girths_all_above, names_all_above = get_cp_z_as_girths_all_above(cp_z_as_girths_all, sail_set)

    # take every second element
    n = 2 * len(sail_set.sails)
    a = tuple(np.array_split(cp_straight_yacht_all, n)[::2])
    cp_straight_yacht_above = np.concatenate(a) 
    
    df_inlet_conditions = np.array_split(myvlm.inlet_conditions.to_df_full(sail_set.sails[0].csys_transformations), 2)
    df_inlet_conditions_above = df_inlet_conditions[0]
    
    df_inviscid_flow = np.array_split(inviscid_flow_results.to_df_full(), 2)
    df_inviscid_flow_above = df_inviscid_flow[0]

    df_cp_straight_yacht_above = pd.DataFrame(cp_straight_yacht_above[:, 2], columns=['cp_points_upright.z'])
    df_cp_z_as_girths_all_above = pd.DataFrame(cp_z_as_girths_all_above, columns=['girths'])
    df_names_all_above = pd.DataFrame(names_all_above, columns=['sail_name'])

    list_of_df = [df_inviscid_flow_above, df_inlet_conditions_above,df_cp_straight_yacht_above, df_cp_z_as_girths_all_above, df_names_all_above ]

    df_inviscid_flow_integral = inviscid_flow_results.to_df_integral()
    df_inlet_conditions_integral = myvlm.inlet_conditions.winds.to_df_integral(sail_set.sails[0].csys_transformations)
    
    file_name = file_name + '.xlsx'
    df_merged = save_to_excel(output_dir, list_of_df, file_name, df_inviscid_flow_integral, df_inlet_conditions_integral)
   
    return df_merged, df_inviscid_flow_integral, df_inlet_conditions_integral
    


def save_to_excel(output_dir : str, list_of_df : List[pd.DataFrame], output_file_name : str, df_inviscid_flow_integral : pd.DataFrame, df_inlet_conditions_integral: pd.DataFrame):
    
    df_merged = pd.concat(list_of_df, axis=1, sort=False)
    max_row = df_merged.shape[0]
    last_col = df_merged.shape[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with pd.ExcelWriter(os.path.join(output_dir, output_file_name)) as writer:
        df_merged.to_excel(writer, startrow=0, sheet_name='Components', float_format="%.8f", index=False)

        workbook = writer.book
        worksheet = writer.sheets['Components']

        def merge_two_dicts(x, y):
            z = x.copy()  # start with keys and values of x
            z.update(y)  # modifies z with keys and values of y
            return z

        #Add some cell formats. https://xlsxwriter.readthedocs.io/example_pandas_column_formats.html
        general_format_dict = {'num_format': '#0.0000',
                               'border': 1,
                               'border_color': '#E8E8E8'  # grey
                               }

        format_general = workbook.add_format(general_format_dict)
        # https://xlsxwriter.readthedocs.io/format.html
        # https://htmlcolorcodes.com/color-picker/
        format_COG = workbook.add_format(
            merge_two_dicts(general_format_dict,
                            {'bg_color': '#FDF8E1'  # light sand
                            }))

        format_COW = workbook.add_format(
            merge_two_dicts(general_format_dict,
                            {'bg_color': '#EAF4FF'  # light sand
                            }))

        format_percent = workbook.add_format(
            {'num_format': '0.00%',
             'border': 1,
             'border_color': '#E8E8E8'  # grey
            })

        for i, col in enumerate(df_merged.columns):
            column_len = df_merged[col].astype(str).str.len().max()
            # Setting the length if the column header is larger than the max column value length
            column_len = max(column_len, len(col)) + 2  # + padding

            worksheet.set_column(i, i, column_len, cell_format=format_general)
            
            if "COG" in col:
                # workaround because set_column finction do not allow to do range cell formatting
                # first row, first col, last row, last col
                worksheet.conditional_format(1, i, max_row, i, {'type': 'cell','criteria': '>=','value': 0, 'format': format_COG})
                worksheet.conditional_format(1, i, max_row, i, {'type': 'cell','criteria': '<','value': 0, 'format': format_COG})

            if "COW" in col:
                worksheet.conditional_format(1, i, max_row, i, {'type': 'cell','criteria': '>=','value': 0, 'format': format_COW})
                worksheet.conditional_format(1, i, max_row, i, {'type': 'cell','criteria': '<','value': 0, 'format': format_COW})

            if col == 'Camber_estimate':
                worksheet.conditional_format(1, i, max_row, i, {'type': 'cell','criteria': '>=','value': 0, 'format': format_percent})
                worksheet.conditional_format(1, i, max_row, i, {'type': 'cell','criteria': '<','value': 0, 'format': format_percent})

        def write_summary_sheet(df, sheet_name):
            df.to_excel(writer, startrow=0, sheet_name=sheet_name, float_format="%.2f", index=False)
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                column_len = df[col].astype(str).str.len().max()
                # Setting the length if the column header is larger than the max column value length
                column_len = max(column_len, len(str(col))) + 0  # + padding
                worksheet.set_column(i, i, column_len, cell_format=format_general)

            for i, row in df.iterrows():
                ri = i + 1
                if "COG" in row['Quantity']:
                    # workaround format first column with string if are not blank (they will always containg string)
                    worksheet.conditional_format(1, 0, ri, 0, {'type':  'no_blanks', 'format': format_COG})
                    worksheet.conditional_format(0, 1, ri, 1, {'type': 'cell','criteria': '>=','value': 0, 'format': format_COG})
                    worksheet.conditional_format(0, 1, ri, 1, {'type': 'cell','criteria': '<','value': 0, 'format': format_COG})

                if "COW" in row['Quantity']:
                    worksheet.conditional_format(1, 0, ri, 0, {'type':  'no_blanks', 'format': format_COW})
                    worksheet.conditional_format(0, 1, ri, 1, {'type': 'cell','criteria': '>=','value': 0, 'format': format_COW})
                    worksheet.conditional_format(0, 1, ri, 1, {'type': 'cell','criteria': '<','value': 0, 'format': format_COW})


        write_summary_sheet(df_inviscid_flow_integral, 'Integrals')
        write_summary_sheet(df_inlet_conditions_integral, 'IC_at_reference_height')

        return df_merged