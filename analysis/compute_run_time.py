import pandas as pd
import os
import numpy as np

total_computing_time = 0

os.chdir('..')
start_path = './hpo_framework/results'
datasets = ['turbofan', 'scania', 'sensor', 'blisk', 'surface']

for _, data_dirs, _ in os.walk(start_path):

    for this_dir in data_dirs:

        if this_dir not in datasets:
            continue
        
        dataset = this_dir
        run_time_this_set = 0

        for _, _, files in os.walk(os.path.join(start_path, this_dir)):

            for this_file in files:

                if 'logs_' in this_file:

                    print('Reading: ', this_file)

                    this_log = pd.read_csv(os.path.join(start_path, this_dir, this_file))

                    for run in this_log['Run-ID'].unique():

                        run_time = this_log.loc[(this_log['Run-ID'] == run), 'timestamps'].max()

                        # print('Time of %s: %f' % (run, run_time))

                        if np.isnan(run_time):
                            continue
                        
                        run_time_this_set += run_time
                        total_computing_time += run_time

        print('Run time on %s data set [s]: %f' % (dataset, run_time_this_set))

print('Total benchmarking time: %f [s]' % total_computing_time)
print('Total benchmarking time: %f [min]' % (total_computing_time/60))
print('Total benchmarking time: %f [h]' % (total_computing_time/60/60))
print('Total benchmarking time: %f [d]' % (total_computing_time/60/60/24))
print('Total benchmarking time: %f [w]' % (total_computing_time/60/60/24/7))