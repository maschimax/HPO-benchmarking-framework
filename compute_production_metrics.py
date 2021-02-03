import os
import pandas as pd
import ast
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_absolute_error, confusion_matrix
from tensorflow import keras
import pickle

from hpo_framework.baseoptimizer import BaseOptimizer
from hpo_framework.trial import Trial
from datasets.Turbofan_Engine_Degradation.turbofan_preprocessing import turbofan_loading_and_preprocessing
from datasets.Scania_APS_Failure.scania_preprocessing import scania_loading_and_preprocessing
from datasets.Sensor_System_Production.sensor_loading_and_balancing import sensor_loading_and_preprocessing
from datasets.Blisk.blisk_preprocessing import blisk_loading_and_preprocessing
from datasets.Surface_Crack_Image.surface_crack_preprocessing import surface_crack_loading_and_preprocessing

initial_path = './hpo_framework/results/'
find_best_configs = False

best_hp_configs = {
    'turbofan': {
        'Max relative improvement': 0.0,
        'ML algorithm': None,
        'HP configuration': None,
        'HPO technique': None,
        'Workers': None,
        'Warm start': None
    },
    'scania': {
        'Max relative improvement': 0.0,
        'ML algorithm': None,
        'HP configuration': None,
        'HPO technique': None,
        'Workers': None,
        'Warm start': None
    },
    'sensor': {
        'Max relative improvement': 0.0,
        'ML algorithm': None,
        'HP configuration': None,
        'HPO technique': None,
        'Workers': None,
        'Warm start': None
    },
    'blisk': {
        'Max relative improvement': 0.0,
        'ML algorithm': None,
        'HP configuration': None,
        'HPO technique': None,
        'Workers': None,
        'Warm start': None
    },
    'surface': {
        'Max relative improvement': 0.0,
        'ML algorithm': None,
        'HP configuration': None,
        'HPO technique': None,
        'Workers': None,
        'Warm start': None
    }
}

new_metrics = {'turbofan': {'MAE': [None, None]},
               'scania': {'Confusion Matrix': [None, None]},
               'sensor': {'Confusion Matrix': [None, None]},
               'blisk': {'MAE': [None, None]},
               'surface': {'Confusion Matrix': [None, None]}}

max_rel_loss_improvement = 0.0
best_ml_algo = None
best_config = None
best_hpo = None
best_workers = None
best_wst = None

if find_best_configs:

    # Find the maximum relative loss improvement for each data set and store the HP configuration, ...
    for this_dataset in best_hp_configs.keys():

        for _, _, result_files in os.walk(os.path.join(initial_path, this_dataset)):

            for this_file in result_files:

                if 'logs_' in this_file:

                    print('Reading log file: ', this_file)
                    log_df = pd.read_csv(os.path.join(
                        initial_path, this_dataset, this_file))

                    print('Log file has %i rows' % log_df.shape[0])
                    print('Searching for HP configuration with max. rel. improvement!')

                    for idx, row in log_df.iterrows():

                        if (idx % 10000) == 0:
                            print(idx)

                        this_best_val_loss = 0
                        this_default_val_loss = 0

                        this_best_val_loss = row['val_losses']
                        this_default_val_loss = row['val_baseline']

                        this_improvement = 100 * \
                            (this_default_val_loss - this_best_val_loss) / \
                            this_default_val_loss

                        if this_improvement > best_hp_configs[this_dataset]['Max relative improvement']:

                            best_hp_configs[this_dataset]['Max relative improvement'] = this_improvement
                            best_hp_configs[this_dataset]['ML algorithm'] = row['ML-algorithm']
                            best_hp_configs[this_dataset]['HP configuration'] = row['configurations']
                            best_hp_configs[this_dataset]['HPO technique'] = row['HPO-method']
                            best_hp_configs[this_dataset]['Workers'] = row['workers']
                            best_hp_configs[this_dataset]['Warm start'] = row['warmstart']

    i = 0
    for this_dataset in best_hp_configs.keys():

        best_hp_configs[this_dataset]['index'] = [i]
        best_hp_configs[this_dataset]['dataset'] = this_dataset

        if i == 0:

            best_config_df = pd.DataFrame.from_dict(
                best_hp_configs[this_dataset])

        else:

            this_config_df = pd.DataFrame.from_dict(
                best_hp_configs[this_dataset])
            best_config_df = pd.concat(
                objs=[best_config_df, this_config_df], axis=0, ignore_index=True)

        i += 1

    best_config_df.to_csv(os.path.join(
        initial_path, 'best_configs_per_dataset.csv'))

config_df = pd.read_csv(os.path.join(
    initial_path, 'best_configs_per_dataset.csv'))

for idx, row in config_df.iterrows():

    dataset = row['dataset']
    ml_algo = row['ML algorithm']

    try:
        hp_config_dict = ast.literal_eval(row['HP configuration'])

    except:
        raise Exception('Wrong HP representation!')

    if dataset == 'turbofan':

        X_train, X_test, y_train, y_test = turbofan_loading_and_preprocessing()
        is_time_series = False

    elif dataset == 'scania':

        X_train, X_test, y_train, y_test = scania_loading_and_preprocessing()
        is_time_series = False

    elif dataset == 'sensor':

        X_train, X_test, y_train, y_test = sensor_loading_and_preprocessing()
        is_time_series = False

    elif dataset == 'blisk':

        X_train, X_test, y_train, y_test = blisk_loading_and_preprocessing()
        is_time_series = True

    elif dataset == 'surface':

        X_train, X_test, y_train, y_test = surface_crack_loading_and_preprocessing()
        is_time_series = False

    else:

        raise Exception('Unknown data set!')

    for this_metric in new_metrics[dataset].keys():

        if this_metric == 'MAE':

            metric = mean_absolute_error

        elif this_metric == 'Confusion Matrix':

            metric = confusion_matrix

        else:

            raise Exception('Unknown metric!')

        print('Computing %s for %s on %s data set!' % (this_metric, ml_algo, dataset))

        dummy_optimizer = BaseOptimizer(hp_space=None, hpo_method=None,
                                        ml_algorithm=ml_algo,
                                        x_train=X_train, x_test=X_test,
                                        y_train=y_train, y_test=y_test,
                                        metric=metric, n_func_evals=1,
                                        random_seed=0, n_workers=1,
                                        cross_val=False, shuffle=False,
                                        is_time_series=is_time_series)

        best_metric_value = dummy_optimizer.train_evaluate_ml_model(params=hp_config_dict,
                                                                    cv_mode=False, test_mode=True)

        dummy_trial = Trial(hp_space=None, ml_algorithm=ml_algo,
                            optimization_schedule=None, metric=metric,
                            n_runs=None, n_func_evals=None, n_workers=None,
                            x_train=X_train, y_train=y_train, x_test=X_test,
                            y_test=y_test, cross_val=False, shuffle=False,
                            is_time_series=is_time_series)

        df_metric_value = dummy_trial.get_baseline(
            cv_mode=False, test_mode=True)

        # Assign results
        new_metrics[dataset][this_metric][1] = best_metric_value
        new_metrics[dataset][this_metric][0] = df_metric_value

with open('new_metrics.p', 'w') as pickle_file:

    pickle.dump(new_metrics, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
