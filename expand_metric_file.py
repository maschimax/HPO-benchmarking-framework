import os
import pandas as pd
import numpy as np
from math import isnan

dataset = 'surface'
file_name = 'metrics_with_cuts_' + dataset + '.csv'
file_path = './hpo_framework/results/' + dataset

metrics_df = pd.read_csv(os.path.join(file_path, file_name))
cols = metrics_df.columns
cols = cols[:-17]

# User's programming ability
# -> low: DefaultValues, Scikit-optimize (GPBO, SMAC), Optuna (TPE, RS, CMA-ES)
# -> medium: HPBandster (BOHB, Hyperband), RoBO (Bohamiann, Fabolas)
# -> high: None
ur_programming_ability = {'Default Values': ['low', 'medium', 'high'],
                          'skopt': ['low', 'medium', 'high'],
                          'optuna': ['low', 'medium', 'high'],
                          'hpbandster': ['medium', 'high'],
                          'robo': ['medium', 'high']}

# UR: Need for model transparency
# -> yes: RandomSearch, DefaultValues
# -> no: remaining HPO techniques
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

# Availability of a well documented library
# -> yes: DefaultValues, HPBandSter (BOHB, Hyperband), Scikit-optimize (GPBO, SMAC), Optuna (TPE, RS, CMA-ES)
# -> no: RoBO (Bohamiann, Fabolas)
ur_well_documented = {'Default Values': ['yes', 'no'],
                      'skopt': ['yes', 'no'],
                      'hpbandster': ['yes', 'no'],
                      'optuna': ['yes', 'no'],
                      'robo': ['no']}

# Pairs of antecedent and crash threshold values
robustness_dict = {'high': 0.25,
                   'low': 1.0}

# Examined 'high' Final Performance use cases only in the benchmarking study
quality_demands = 'high'

expanded_df = pd.DataFrame(columns=cols)

print('Expanding metrics file with ' + str(len(metrics_df)) + ' lines!')

for idx, row in metrics_df.iterrows():

    print('Expanding row # ', idx)

    this_hpo = row['HPO-method']
    this_lib = row['HPO-library']

    # Iterate over programming abilities
    for this_ability in ['low', 'medium', 'high']:

        # Iterate over transparency categories
        for this_transparency in ['yes', 'no']:

            # Iterate over documentation categories
            for this_doc in ['yes', 'no']:

                # Iterate over robustness category
                for this_robustness_category in robustness_dict.keys():

                    this_crash_share = row['Crashes'] / row['Runs']

                    # Check, whether this HPO technique is suitable based on the URs, etc.
                    if this_ability not in ur_programming_ability[this_lib] \
                            or this_transparency not in ur_transparency[this_hpo] \
                            or this_doc not in ur_well_documented[this_lib] \
                            or this_crash_share > robustness_dict[this_robustness_category]:

                        # The HPO technique is not suitable for this use case (at least on requirement is not met
                        hpo_suitability = 'no'

                    else:

                        # The HPO technique is suitable for this use case -> all requirements are met
                        hpo_suitability = 'yes'

                    uc_id = "%s-%s-%s-%s-%s-%s-%s-%s" % (dataset, row['ML-algorithm'], row['Workers'],
                                                         row['Warmstart'], this_ability, this_transparency, this_doc, this_robustness_category)

                    if not np.isnan(row['Max Cut Time Budget [s]']):
                        max_cut_row = row.loc[cols]
                        max_cut_row['Wall clock time [s]'] = row.loc['Max Cut Time Budget [s]']
                        if 'Max Cut Test Loss' in row:
                            max_cut_row['Mean (final test loss)'] = row.loc['Max Cut Test Loss']
                        else:
                            max_cut_row['Mean (final test loss)'] = np.nan
                        max_cut_row['Mean (final validation loss)'] = row.loc['Max Cut Validation Loss']
                        max_cut_row['Area under curve (AUC)'] = row.loc['Max Cut AUC']
                        max_cut_row['Test loss ratio (default / best)'] = np.nan
                        max_cut_row['Interquartile range (final test loss)'] = np.nan
                        max_cut_row['Generalization error'] = np.nan
                        max_cut_row['Evaluations for best configuration'] = np.nan
                        max_cut_row["User's programming ability"] = this_ability
                        max_cut_row['UR: need for model transparency'] = this_transparency
                        max_cut_row['UR: Availability of a well documented library'] = this_doc
                        max_cut_row['Robustness'] = this_robustness_category
                        max_cut_row['UR: quality demands'] = quality_demands
                        max_cut_row['HPO suitability'] = hpo_suitability

                        this_uc_id = uc_id + '-' + \
                            str(round(row['Wall clock time [s]'], 2))
                        max_cut_row['ID'] = this_uc_id

                        expanded_df = expanded_df.append(
                            max_cut_row, ignore_index=True)

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
                        second_cut_row['UR: need for model transparency'] = this_transparency
                        second_cut_row['UR: Availability of a well documented library'] = this_doc
                        second_cut_row['Robustness'] = this_robustness_category
                        second_cut_row['UR: quality demands'] = quality_demands
                        second_cut_row['HPO suitability'] = hpo_suitability

                        this_uc_id = uc_id + '-' + \
                            str(round(row['2nd Cut Time Budget [s]'], 2))
                        second_cut_row['ID'] = this_uc_id

                        expanded_df = expanded_df.append(
                            second_cut_row, ignore_index=True)

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
                        third_cut_row['UR: need for model transparency'] = this_transparency
                        third_cut_row['UR: Availability of a well documented library'] = this_doc
                        third_cut_row['Robustness'] = this_robustness_category
                        third_cut_row['UR: quality demands'] = quality_demands
                        third_cut_row['HPO suitability'] = hpo_suitability

                        this_uc_id = uc_id + '-' + \
                            str(round(row['3rd Cut Time Budget [s]'], 2))
                        third_cut_row['ID'] = this_uc_id

                        expanded_df = expanded_df.append(
                            third_cut_row, ignore_index=True)

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
                        fourth_cut_row['UR: need for model transparency'] = this_transparency
                        fourth_cut_row['UR: Availability of a well documented library'] = this_doc
                        fourth_cut_row['Robustness'] = this_robustness_category
                        fourth_cut_row['UR: quality demands'] = quality_demands
                        fourth_cut_row['HPO suitability'] = hpo_suitability

                        this_uc_id = uc_id + '-' + \
                            str(round(row['4th Cut Time Budget [s]'], 2))
                        fourth_cut_row['ID'] = this_uc_id

                        expanded_df = expanded_df.append(
                            fourth_cut_row, ignore_index=True)

expanded_df.drop(['Unnamed: 0'], axis=1, inplace=True)

print('The expanded file has ' + str(len(expanded_df)) + ' lines!')

exp_filename = 'expanded_metrics_' + dataset + '.csv'
expanded_df.to_csv(os.path.join(file_path, exp_filename))
