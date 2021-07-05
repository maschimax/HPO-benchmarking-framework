import os
import pandas as pd

# Specify the data set
dataset = 'blisk'
bm_dir = 'C:/Users/Max/OneDrive - rwth-aachen.de/Uni/Master/Masterarbeit/01_Content/05_Benchmarking Study/' + dataset

# Count the number of log files
log_count = 0

for _, dirs, _ in os.walk(bm_dir):

    # Iterate over all sub-folders of the directory
    for this_dir in dirs:

        # Jump into the sub-folders
        for _, log_dirs, _ in os.walk(os.path.join(bm_dir, this_dir)):

            # Iterate over all files in the sub_folder
            for log_dir in log_dirs:

                print(log_dir)

                # Jump into the log-folders
                for _, _, log_files in os.walk(os.path.join(bm_dir, this_dir, log_dir)):

                    # Iterate over all log files in the log-folder
                    for log in log_files:

                        # Read the log file to pd.DataFrame
                        print('Reading: ', log)
                        log_df = pd.read_csv(os.path.join(bm_dir, this_dir, log_dir, log), index_col=0)

                        if log_count == 0:
                            aggregated_log_df = log_df.copy(deep=True)
                        else:
                            aggregated_log_df = pd.concat(objs=[aggregated_log_df, log_df], axis=0, ignore_index=True)

                        log_count += 1

# Store them in a single DataFrame / .csv-file
os.chdir('..')
save_dir = './hpo_framework/results/' + dataset

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

file_name = 'logs_' + dataset + '.csv'
file_path = os.path.join(save_dir, file_name)

# Save aggregated pd.DataFrame to .csv-file
aggregated_log_df.to_csv(path_or_buf=file_path)
