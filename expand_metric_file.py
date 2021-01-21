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

# TODO: Expand by UR and Robustness -> i.e. Filter non robust HPO-techniques here?
# TODO: Avoid duplicates here?

# User's programming ability -> low: DefaultValues, Scikit-optimize (GPBO, SMAC), Optuna (TPE, RS, CMA-ES) //
# medium: HPBandster (BOHB, Hyperband), RoBO (Bohamiann, Fabolas) // high: None
ur_programming_ability = {'Default Values': ['low', 'medium', 'high'],
                          'skopt': ['low', 'medium', 'high'],
                          'optuna': ['low', 'medium', 'high'],
                          'hpbandster': ['medium', 'high'],
                          'robo': ['medium', 'high']}

# UR: Need for model transparency -> yes: RandomSearch, DefaultValues // no: remaining
ur_transparency = {'Default Values': ['yes', 'no'],
                   'RandomSearch': ['yes', 'no'],
                   'GPBO': ['no'],
                   'SMAC': ['no'],
                   'TPE': ['no'],
                   'CMA-ES': ['no'],
                   'Hyperband': ['no'],
                   'BOHB': ['no'],
                   'Fabolas': ['no'],
                   'Bohamiann': ['no']}

# Availability of a well documented library -> yes: DefaultValues, HPBandSter (BOHB, Hyperband),
# Scikit-optimize (GPBO, SMAC), Optuna (TPE, RS, CMA-ES) // no: RoBO (Bohamiann, Fabolas)
ur_well_documented = {'Default Values': ['yes', 'no'],
                      'skopt': ['yes', 'no'],
                      'hpbandster': ['yes', 'no'],
                      'optuna': ['yes', 'no'],
                      'robo': ['no']}

# Pairs of antecedent and crash threshold values
robustness_dict = {'high': 0.5,
                   'low': 1.0}

expanded_df = pd.DataFrame(columns=cols)

print('Expanding metrics file with ' + str(len(metrics_df)) + ' lines!')

for idx, row in metrics_df.iterrows():

    print('Expanding row # ', idx)

    this_hpo = row['HPO-method']
    this_lib = row['HPO-library']

    for this_ability in ur_programming_ability[this_lib]:

        for this_transparency in ur_transparency[this_hpo]:

            for this_doc in ur_well_documented[this_lib]:

                for this_robustness_category in robustness_dict.keys():

                        this_crash_share = row['Crashes'] / row['Runs']

                        if this_crash_share > robustness_dict[this_robustness_category]:
                            continue

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
                            max_cut_row["User's programming ability"] = this_ability
                            max_cut_row['UR: Need for model transparency'] = this_transparency
                            max_cut_row['UR: Availability of a well documented library'] = this_doc
                            max_cut_row['Robustness'] = this_robustness_category

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
                            second_cut_row["User's programming ability"] = this_ability
                            second_cut_row['UR: Need for model transparency'] = this_transparency
                            second_cut_row['UR: Availability of a well documented library'] = this_doc
                            second_cut_row['Robustness'] = this_robustness_category

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
                            third_cut_row["User's programming ability"] = this_ability
                            third_cut_row['UR: Need for model transparency'] = this_transparency
                            third_cut_row['UR: Availability of a well documented library'] = this_doc
                            third_cut_row['Robustness'] = this_robustness_category

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
                            fourth_cut_row["User's programming ability"] = this_ability
                            fourth_cut_row['UR: Need for model transparency'] = this_transparency
                            fourth_cut_row['UR: Availability of a well documented library'] = this_doc
                            fourth_cut_row['Robustness'] = this_robustness_category

                            expanded_df = expanded_df.append(fourth_cut_row, ignore_index=True)

expanded_df.drop(['Unnamed: 0'], axis=1, inplace=True)

print('The expanded file has ' + str(len(expanded_df)) + ' lines!')

exp_filename = 'expanded_metrics_' + dataset + '.csv'
expanded_df.to_csv(os.path.join(file_path, exp_filename))
