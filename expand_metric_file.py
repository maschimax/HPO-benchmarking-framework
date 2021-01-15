import os
import pandas as pd
import numpy as np
from math import isnan

dataset = 'turbofan'
file_name = 'metrics_with_cuts_' + dataset + '.csv'
file_path = './hpo_framework/results/' + dataset

metrics_df = pd.read_csv(os.path.join(file_path, file_name))
cols = metrics_df.columns
cols = cols[:-17]

expanded_df = pd.DataFrame(columns=cols)

for idx, row in metrics_df.iterrows():
    if not np.isnan(row['Max Cut Time Budget [s]']):
        max_cut_row = row.loc[cols]
        max_cut_row['Wall clock time [s]'] = row.loc['Max Cut Time Budget [s]']
        max_cut_row['Mean (final test loss)'] = row.loc['Max Cut Test Loss']
        max_cut_row['Mean (final validation loss)'] = row.loc['Max Cut Validation Loss']
        max_cut_row['Area under curve (AUC)'] = row.loc['Max Cut AUC']
        max_cut_row['Test loss ratio (default / best)'] = np.nan
        max_cut_row['Interquartile range (final test loss)'] = np.nan
        max_cut_row['Generalization error'] = np.nan
        max_cut_row['Evaluations for best configuration'] = np.nan

        expanded_df = expanded_df.append(max_cut_row, ignore_index=True)

    if not np.isnan(row['2nd Cut Time Budget [s]']):
        second_cut_row = row.loc[cols]
        second_cut_row['Wall clock time [s]'] = row.loc['2nd Cut Time Budget [s]']
        second_cut_row['Mean (final test loss)'] = np.nan
        second_cut_row['Mean (final validation loss)'] = row.loc['2nd Cut Validation Loss']
        second_cut_row['Area under curve (AUC)'] = row.loc['2nd Cut AUC']
        second_cut_row['Test loss ratio (default / best)'] = np.nan
        second_cut_row['Interquartile range (final test loss)'] = np.nan
        second_cut_row['Generalization error'] = np.nan
        second_cut_row['Evaluations for best configuration'] = np.nan
        second_cut_row['Evaluations'] = np.nan

        expanded_df = expanded_df.append(second_cut_row, ignore_index=True)
    if not np.isnan(row['3rd Cut Time Budget [s]']):
        third_cut_row = row.loc[cols]
        third_cut_row['Wall clock time [s]'] = row.loc['3rd Cut Time Budget [s]']
        third_cut_row['Mean (final test loss)'] = np.nan
        third_cut_row['Mean (final validation loss)'] = row.loc['3rd Cut Validation Loss']
        third_cut_row['Area under curve (AUC)'] = row.loc['3rd Cut AUC']
        third_cut_row['Test loss ratio (default / best)'] = np.nan
        third_cut_row['Interquartile range (final test loss)'] = np.nan
        third_cut_row['Generalization error'] = np.nan
        third_cut_row['Evaluations for best configuration'] = np.nan
        third_cut_row['Evaluations'] = np.nan

        expanded_df = expanded_df.append(third_cut_row, ignore_index=True)

    if not np.isnan(row['4th Cut Time Budget [s]']):
        fourth_cut_row = row.loc[cols]
        fourth_cut_row['Wall clock time [s]'] = row.loc['4th Cut Time Budget [s]']
        fourth_cut_row['Mean (final test loss)'] = np.nan
        fourth_cut_row['Mean (final validation loss)'] = row.loc['4th Cut Validation Loss']
        fourth_cut_row['Area under curve (AUC)'] = row.loc['4th Cut AUC']
        fourth_cut_row['Test loss ratio (default / best)'] = np.nan
        fourth_cut_row['Interquartile range (final test loss)'] = np.nan
        fourth_cut_row['Generalization error'] = np.nan
        fourth_cut_row['Evaluations for best configuration'] = np.nan
        fourth_cut_row['Evaluations'] = np.nan

        expanded_df = expanded_df.append(third_cut_row, ignore_index=True)

expanded_df.drop(['Unnamed: 0'], axis=1, inplace=True)

exp_filename = 'expanded_metrics_' + dataset + '.csv'
expanded_df.to_csv(os.path.join(file_path, exp_filename))
