import os
import pandas as pd

search_path = './hpo_framework/results'

max_rel_loss_improvement = 0.0
best_hpo = None
ml_algo = None
dataset = None
workers = None
warm_start = None
default_loss = None
hpo_loss = None

for _, data_dirs, _ in os.walk(search_path):

    for this_dataset in data_dirs:

        for _, _, files in os.walk(os.path.join(search_path, this_dataset)):

            for this_file in files:

                if 'metrics_with_cuts' in this_file:

                    print('Reading: ', this_file)

                    this_metric_df = pd.read_csv(os.path.join(search_path, this_dataset, this_file), index_col=0)

                    for idx, row in this_metric_df.iterrows():

                        this_best_val_loss = 0
                        this_default_val_loss = 0

                        this_best_val_loss = row['Mean (final validation loss)']
                        this_default_val_loss = row['Validation baseline']

                        this_improvement = 100 * (this_default_val_loss - this_best_val_loss) / this_default_val_loss

                        if this_improvement > max_rel_loss_improvement:

                            max_rel_loss_improvement = this_improvement
                            best_hpo = row['HPO-method']
                            ml_algo = row['ML-algorithm']
                            dataset = this_dataset
                            workers = row['Workers']
                            warm_start = row['Warmstart']
                            default_loss = this_default_val_loss
                            hpo_loss = this_best_val_loss



print('Max. loss improvement [%]: ', max_rel_loss_improvement)
print('HPO technique: ', best_hpo)
print('ML algorithm: ', ml_algo)
print('Data set: ', dataset)
print('Workers: ', workers)
print('Warm start: ', warm_start)
print('DV loss: ', default_loss)
print('HPO loss: ', hpo_loss)
