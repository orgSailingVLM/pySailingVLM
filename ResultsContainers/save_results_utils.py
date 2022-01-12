import pandas as pd
import os

from ResultsContainers.InviscidFlowResults import InviscidFlowResults
# from LLT_optimizer.SectionShapeResults import SectionShapeResults
from Inlet.InletConditions import InletConditions
from YachtGeometry.SailGeometry import SailSet


def save_results_to_file(inviscid_flow_results: InviscidFlowResults,
                         section_results,
                         inlet_conditions: InletConditions,
                         sail_set: SailSet,
                         output_dir="output"):

    girths_as_dict = {'girths': sail_set.sail_cp_to_girths()}
    df_girths = sail_set.extract_data_above_water_to_df(pd.DataFrame.from_records(girths_as_dict))
    df_sail_names = sail_set.extract_data_above_water_to_df(sail_set.get_sail_name_for_each_element())

    cp_points_upright_as_dict = {'cp_points_upright.z': sail_set.get_cp_points_upright()[:, 2]}
    df_cp_points_upright = sail_set.extract_data_above_water_to_df(pd.DataFrame.from_records(cp_points_upright_as_dict))

    df_inlet_conditions = sail_set.extract_data_above_water_to_df(inlet_conditions.to_df_full(sail_set.sails[0].csys_transformations))
    df_inviscid_flow = sail_set.extract_data_above_water_to_df(inviscid_flow_results.to_df_full())
    list_of_df = [df_inviscid_flow, df_inlet_conditions, df_cp_points_upright, df_girths, df_sail_names]

    if section_results is not None:
        df_section_results = sail_set.extract_data_above_water_to_df(section_results.to_df_full())
        list_of_df.append(df_section_results)

    df_merged = pd.concat(list_of_df, axis=1, sort=False)
    df_inviscid_flow_integral = inviscid_flow_results.to_df_integral()

    # if isinstance(inlet_conditions.winds, ExpWindProfile):
    df_inlet_conditions_integral = inlet_conditions.winds.to_df_integral(sail_set.sails[0].csys_transformations)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with pd.ExcelWriter(os.path.join(output_dir, "Results.xlsx")) as writer:
        df_merged.to_excel(writer, startrow=0, sheet_name='Components', float_format="%.4f", index=False)

        workbook = writer.book
        worksheet = writer.sheets['Components']

        def merge_two_dicts(x, y):
            z = x.copy()  # start with keys and values of x
            z.update(y)  # modifies z with keys and values of y
            return z

        # Add some cell formats. https://xlsxwriter.readthedocs.io/example_pandas_column_formats.html
        # format1 = workbook.add_format({'num_format': '#,##0.0000'})
        general_format_dict = {'num_format': '#0.00',
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
                worksheet.set_column(i, i, column_len, cell_format=format_COG)

            if "COW" in col:
                worksheet.set_column(i, i, column_len, cell_format=format_COW)

            if col == 'Camber_estimate':
                worksheet.set_column(i, i, column_len, cell_format=format_percent)

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
                    worksheet.set_row(ri, cell_format=format_COG)

                if "COW" in row['Quantity']:
                    worksheet.set_row(ri, cell_format=format_COW)

        write_summary_sheet(df_inviscid_flow_integral, 'Integrals')
        write_summary_sheet(df_inlet_conditions_integral, 'IC_at_reference_height')

    return df_merged, df_inviscid_flow_integral, df_inlet_conditions_integral
