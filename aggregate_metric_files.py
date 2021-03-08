import os
import pandas as pd

dataset = 'blisk'
dir = 'C:/Users/Max/OneDrive - rwth-aachen.de/Uni/Master/Masterarbeit/01_Content/05_Benchmarking Study/' + dataset

metrics_count = 0

for _, dirs, _ in os.walk(dir):

    # iterate over all sub-folders of the directory
    for this_dir in dirs:

        # Iterate over all files in the sub_folder
        for _, _, result_files in os.walk(os.path.join(dir, this_dir)):

            for this_file in result_files:

                # Check, whether the file is a metrics file
                if 'metrics_' in this_file:

                    # Load the metrics file as a pd.DataFrame
                    print('reading: ' + this_file)
                    metrics_df = pd.read_csv(os.path.join(dir, this_dir, this_file), index_col=0)

                    # If first file -> copy, else concatenate
                    if metrics_count == 0:
                        aggregated_df = metrics_df
                    else:
                        aggregated_df = pd.concat(objs=[aggregated_df, metrics_df], axis=0)

                    metrics_count += 1

                else:
                    continue

# Reset index
aggregated_df.reset_index(drop=True, inplace=True)

# Save aggregated metrics df as a .csv-File
abs_results_path = os.path.abspath(path='./hpo_framework/results/' + dataset)

if not os.path.isdir(abs_results_path):
    os.mkdir(abs_results_path)

filename = 'metrics_' + dataset + '.csv'
file_path = os.path.join(abs_results_path, filename)

aggregated_df.to_csv(file_path)
