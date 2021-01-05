import pandas as pd
import numpy as np
import ast

from hpo_framework.baseoptimizer import BaseOptimizer
from datasets.Turbofan_Engine_Degradation.turbofan_preprocessing import turbofan_loading_and_preprocessing
from datasets.Scania_APS_Failure.scania_preprocessing import scania_loading_and_preprocessing
from datasets.Sensor_System_Production.sensor_loading_and_balancing import sensor_loading_and_preprocessing
from datasets.Blisk.blisk_preprocessing import blisk_loading_and_preprocessing
from hpo_framework import hpo_metrics

dataset = 'turbofan'
find_budget_and_retrain = True
write_to_metrics = True

# Setup variants (# workers, warm start (Yes / No))
setup_vars = [(1, False), (8, False), (1, True)]

if find_budget_and_retrain:

    ####################################################################################################################
    # PART 1: Find time budgets for each use case (minimum wall clock time)

    log_path = './hpo_framework/results/' + dataset + '/logs_' + dataset + '.csv'

    # Read the log file -> pd.DataFrame
    log_df = pd.read_csv(log_path, index_col=0)

    worker_list = []
    warm_start_list = []
    algo_list = []
    hpo_tech_list = []
    min_time_list = []

    # Iterate over setup variants
    for this_setup in setup_vars:

        # Filter by setup
        setup_df = log_df.loc[(log_df['workers'] == this_setup[0]) & (log_df['warmstart'] == this_setup[1]), :]

        # ML algorithms of this setup variant
        ml_algos = list(setup_df['ML-algorithm'].unique())

        # Iterate over ML-algorithms
        for this_algo in ml_algos:

            # Filter by ML algorithm -> use case
            use_case_df = setup_df.loc[setup_df['ML-algorithm'] == this_algo, :]

            # Initialize list for saving tuples of HPO-techniques, Run ID and wall clock times
            time_budget_list = []

            hpo_techs = list(use_case_df['HPO-method'].unique())

            # Iterate over HPO-techniques
            for this_tech in hpo_techs:

                # Filter by HPO-technique
                hpo_df = use_case_df.loc[use_case_df['HPO-method'] == this_tech, :]

                # Run IDs
                runs = list(hpo_df['Run-ID'].unique())

                # Iterate over runs per HPO-technique
                for this_run in runs:

                    # Filter by run
                    run_df = hpo_df.loc[hpo_df['Run-ID'] == this_run]

                    run_successful = run_df['run_successful'].to_numpy()[0]

                    if not run_successful:
                        continue

                    # Find maximum timestamp of this run (-> when the full evaluation budget was consumed)
                    this_wc_time = np.nanmax(run_df['timestamps'].to_numpy())

                    # Append results
                    time_budget_list.append((this_tech, this_run, this_wc_time))

            # Determine the 'fastest' HPO technique with the minimal wall clock time for this use case
            time_lb = np.float('inf')
            fastest_tech = None

            for time_tuple in time_budget_list:
                if time_tuple[2] < time_lb:
                    time_lb = time_tuple[2]
                    fastest_tech = time_tuple[0]

            # Append results
            worker_list.append(this_setup[0])
            warm_start_list.append(this_setup[1])
            algo_list.append(this_algo)
            hpo_tech_list.append(fastest_tech)
            min_time_list.append(time_lb)

            print('ML algorithm: ', this_algo)
            print('Time budget: ', time_lb)
            print('Fastest HPO technique: ', fastest_tech)
            print('-------------------------------------')

    # Create pd.DataFrame to store the results
    time_budget_df = pd.DataFrame({
        'ML-algorithm': algo_list,
        'Workers': worker_list,
        'Warm start': warm_start_list,
        'Time budget [s]': min_time_list,
        'Fastest HPO-technique': hpo_tech_list
    })

    # Write results to .csv-file
    tb_path = './hpo_framework/results/' + dataset + '/time_budgets_' + dataset + '.csv'
    time_budget_df.to_csv(tb_path)

    ####################################################################################################################
    # PART 2: Retrain the ML algorithms

    trial_id_list = []
    algo_list = []
    hpo_tech_list = []
    worker_list = []
    warm_start_list = []
    budget_list = []
    tbloss_list = []
    fast_tech_list = []

    data_is_loaded = False

    # Iterate over setup variants
    for this_setup in setup_vars:

        # Filter by setup
        setup_df = log_df.loc[(log_df['workers'] == this_setup[0]) & (log_df['warmstart'] == this_setup[1]), :]

        # ML algorithms of this setup variant
        ml_algos = list(setup_df['ML-algorithm'].unique())

        # Iterate over ML-algorithms
        for this_algo in ml_algos:

            # Time budget for this use case
            this_budget = time_budget_df.loc[(time_budget_df['ML-algorithm'] == this_algo) &
                                             (time_budget_df['Workers'] == this_setup[0]) &
                                             (time_budget_df['Warm start'] == this_setup[1]),
                                             'Time budget [s]'].values[0]

            this_fastest_tech = time_budget_df.loc[(time_budget_df['ML-algorithm'] == this_algo) &
                                                   (time_budget_df['Workers'] == this_setup[0]) &
                                                   (time_budget_df['Warm start'] == this_setup[1]),
                                                   'Fastest HPO-technique'].values[0]

            # Filter by ML algorithm -> use case
            use_case_df = setup_df.loc[setup_df['ML-algorithm'] == this_algo, :]

            hpo_techs = list(use_case_df['HPO-method'].unique())

            # Iterate over HPO-techniques
            for this_tech in hpo_techs:

                # Filter by HPO-technique
                hpo_df = use_case_df.loc[use_case_df['HPO-method'] == this_tech, :]

                # Run IDs
                runs = list(hpo_df['Run-ID'].unique())

                test_loss_list = []

                # Iterate over runs per HPO-technique
                for this_run in runs:

                    # Filter by run
                    run_df = hpo_df.loc[hpo_df['Run-ID'] == this_run]

                    run_successful = run_df['run_successful'].to_numpy()[0]

                    if not run_successful:
                        continue

                    # Index of minimum validation loss achieved within the time budget
                    min_val_loss_idx = run_df.loc[run_df['timestamps'] <= this_budget,
                                                  'val_losses'].idxmin(axis=0, skipna=True)

                    # Best found HP configuration within the time budget
                    try:
                        best_config_dict = ast.literal_eval(run_df.loc[min_val_loss_idx, 'configurations'])
                    except:  # Very messy error handling
                        print('-----------------------------------------------------')
                        print('Cleaning wrong HP representation in log file!')

                        if this_algo == 'AdaBoostRegressor' or this_algo == 'AdaBoostClassifier':
                            config_str = run_df.loc[min_val_loss_idx, 'configurations']
                            mem_dict = dict((x.strip('{'), y.strip('}')) for x, y in (element.split(':') for element
                                                                                      in config_str.split(', ')))

                            clean_dict = {}
                            for old_key in mem_dict.keys():
                                new_key = old_key.strip("'")
                                clean_dict[new_key] = mem_dict[old_key]

                            clean_dict['n_estimators'] = int(clean_dict['n_estimators'])
                            clean_dict['learning_rate'] = float(clean_dict['learning_rate'])
                            clean_dict['loss'] = clean_dict['loss'].strip().strip("'")

                            _, max_depth = clean_dict['base_estimator'].split('=')
                            max_depth = max_depth.strip().strip(')')
                            del clean_dict['base_estimator']
                            clean_dict['max_depth'] = int(max_depth)

                            print('Cleaned HP representation: ', clean_dict)

                            best_config_dict = clean_dict
                        else:
                            raise Exception('Error handling required!')

                    # Retrain the ML algorithm for this hyperparameter configuration and calculate the test loss
                    loss_metric = run_df['loss_metric'].unique()[0]
                    this_seed = run_df['random_seed'].unique()[0]

                    # Load the data set
                    if dataset == 'turbofan' and not data_is_loaded:
                        do_shuffle = True
                        X_train, X_test, y_train, y_test = turbofan_loading_and_preprocessing()
                        data_is_loaded = True
                        is_time_series = False

                    elif dataset == 'scania' and not data_is_loaded:
                        do_shuffle = True
                        X_train, X_test, y_train, y_test = scania_loading_and_preprocessing()
                        data_is_loaded = True
                        is_time_series = False

                    elif dataset == 'sensor' and not data_is_loaded:
                        do_shuffle = True
                        X_train, X_test, y_train, y_test = sensor_loading_and_preprocessing()
                        data_is_loaded = True
                        is_time_series = False

                    elif dataset == 'blisk' and not data_is_loaded:
                        do_shuffle = False
                        X_train, X_test, y_train, y_test = blisk_loading_and_preprocessing()
                        data_is_loaded = True
                        is_time_series = True

                    elif data_is_loaded:
                        raise Exception('Unknown data set!')

                    # Select the loss metric
                    if loss_metric == 'RUL-loss':

                        this_metric = hpo_metrics.rul_loss_score

                    elif loss_metric == 'root_mean_squared_error':

                        this_metric = hpo_metrics.root_mean_squared_error

                    elif loss_metric == 'F1-loss':

                        this_metric = hpo_metrics.f1_loss

                    else:
                        raise Exception('Unknown loss metric!')

                    dummy_optimizer = BaseOptimizer(hp_space=None, hpo_method=None, ml_algorithm=this_algo,
                                                    x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test,
                                                    metric=this_metric, n_func_evals=1,
                                                    random_seed=this_seed, n_workers=this_setup[0], cross_val=False,
                                                    shuffle=do_shuffle, is_time_series=is_time_series)

                    print('-----------------------------------------------------')
                    print('Retrain ' + this_algo + ' on ' + dataset + ' data set.')
                    test_loss = dummy_optimizer.train_evaluate_ml_model(params=best_config_dict, cv_mode=False,
                                                                        test_mode=True)

                    print('Test loss: ', test_loss)

                    test_loss_list.append(test_loss)

                # Calculate the mean test loss over all runs of this HPO-technique
                test_loss_arr = np.array(test_loss_list)
                mean_test_loss = np.nanmean(test_loss_arr)

                # Append results
                trial_id_list.append(hpo_df['Trial-ID'].unique()[0])
                algo_list.append(this_algo)
                hpo_tech_list.append(this_tech)
                worker_list.append(this_setup[0])
                warm_start_list.append(this_setup[1])
                budget_list.append(this_budget)
                tbloss_list.append(mean_test_loss)
                fast_tech_list.append(this_fastest_tech)

    # Create a pd.DataFrame to store the results
    tb_loss_df = pd.DataFrame({
        'Trial-ID': trial_id_list,
        'ML-algorithm': algo_list,
        'HPO-method': hpo_tech_list,
        'Workers': worker_list,
        'Warm start': warm_start_list,
        'Time Budget [s]': budget_list,
        'Min. test loss in time budget': tbloss_list,
        'Fastest HPO-technique': fast_tech_list
    })

    # Write results to .csv-file
    tb_loss_path = './hpo_framework/results/' + dataset + '/min_losses_in_time_budget_' + dataset + '.csv'
    tb_loss_df.to_csv(tb_loss_path)

########################################################################################################################


if write_to_metrics:
    # PART 3: Write the results to the metrics file
    metrics_path = './hpo_framework/results/' + dataset + '/metrics_' + dataset + '.csv'
    tb_loss_path = './hpo_framework/results/' + dataset + '/min_losses_in_time_budget_' + dataset + '.csv'

    # Load .csv-files
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    tb_df = pd.read_csv(tb_loss_path, index_col=0)

    # Count the number of modified lines in the metrics file
    col_count = 0

    # Iterate through the rows in the tb_df
    for idx, row in tb_df.iterrows():
        # Query results
        trial_id = row['Trial-ID']
        min_test_loss = row['Min. test loss in time budget']
        time_budget = row['Time Budget [s]']
        fast_tech = row['Fastest HPO-technique']

        # Write results to metrics_df
        metrics_df.loc[metrics_df['Trial-ID'] == trial_id, 'Time Budget [s]'] = time_budget
        metrics_df.loc[metrics_df['Trial-ID'] == trial_id, 'Min. avg. test loss in time budget'] = min_test_loss
        metrics_df.loc[metrics_df['Trial-ID'] == trial_id, 'Fastest HPO-Technique'] = fast_tech

        print(str(col_count) + ' - Modified Trial: ' + trial_id)
        col_count += 1

    print(str(col_count) + ' of ' + str(len(metrics_df)) + ' total lines have been modified!')

    # Save modified metrics .csv-file
    metrics_df.to_csv(metrics_path)
