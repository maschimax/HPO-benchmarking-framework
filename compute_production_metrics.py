import os
import pandas as pd
import ast
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_absolute_error, confusion_matrix
import json

from hpo_framework.baseoptimizer import BaseOptimizer
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

    elif dataset == 'scania':

        X_train, X_test, y_train, y_test = scania_loading_and_preprocessing()

    elif dataset == 'sensor':

        X_train, X_test, y_train, y_test = sensor_loading_and_preprocessing()

    elif dataset == 'blisk':

        X_train, X_test, y_train, y_test = blisk_loading_and_preprocessing()

    elif dataset == 'surface':

        X_train, X_test, y_train, y_test = surface_crack_loading_and_preprocessing()
        
    else:

        raise Exception('Unknown data set!')

    for this_metric in new_metrics[dataset].keys():

        if this_metric == 'MAE':

            metric = mean_absolute_error

        elif this_metric == 'Confusion Matrix':

            metric = confusion_matrix

        else:

            raise Exception('Unknown metric!')
            

        if ml_algo == 'DecisionTreeRegressor':

            # Inialize models with default and best HPs
            df_model = DecisionTreeRegressor(random_state=0)
            best_model = DecisionTreeRegressor(**hp_config_dict, random_state=0)

        elif ml_algo == 'AdaBoostClassifier':
            
            # Inialize models with default and best HPs
            df_model = AdaBoostClassifier(random_state=0)

            max_depth = hp_config_dict.pop('max_depth')
            hp_config_dict['base_estimator'] = DecisionTreeClassifier(max_depth=max_depth)
            best_model = AdaBoostClassifier(**hp_config_dict, random_state=0)

        elif ml_algo == 'KerasClassifier':

            

        else:

            print(ml_algo)
            raise Exception('Unknown ML algorithm!')

        # Train best model
        best_model.fit(X_train, y_train)
        best_y_pred = best_model.predict(X_test)
        best_loss = metric(y_test, best_y_pred)

        # Train default model
        df_model.fit(X_train, y_train)
        df_y_pred = df_model.predict(X_test)
        df_loss = metric(y_test, df_y_pred)

        # Assign results
        new_metrics[dataset][this_metric][0] = df_loss
        new_metrics[dataset][this_metric][1] = best_loss

with open('new_metrics.json', 'w') as js_file:

    json.dump(new_metrics, js_file)
