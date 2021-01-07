import pandas as pd
import numpy as np
import ast
import uuid

from hpo_framework.baseoptimizer import BaseOptimizer
from datasets.Turbofan_Engine_Degradation.turbofan_preprocessing import turbofan_loading_and_preprocessing
from datasets.Scania_APS_Failure.scania_preprocessing import scania_loading_and_preprocessing
from datasets.Sensor_System_Production.sensor_loading_and_balancing import sensor_loading_and_preprocessing
from datasets.Blisk.blisk_preprocessing import blisk_loading_and_preprocessing
from hpo_framework import hpo_metrics

def find_time_budget(log_df: pd.DataFrame):
    """
    Identify the time budget for each use case (wall clock time of fastest optimization run)
    :return: time_budget_df
    """

    worker_list = []
    warm_start_list = []
    algo_list = []
    hpo_tech_list = []
    min_time_list = []

    # Setup variants (# workers, warm start (Yes / No))
    setup_vars = [(1, False), (8, False), (1, True)]

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

    return time_budget_df


def get_losses_in_time_budget(time_budget_df: pd.DataFrame, log_df: pd.DataFrame, compute_test_loss=True,
                              new_use_case=False):
    """
    The function computes the validation and (if compute_test_loss=True) the test loss, that has been achieved by the
     HPO techniques within the time budget.
    :param time_budget_df: DataFrame containing a time budget for each BM use case for this data set.
    :param log_df: DataFrame, that stores the log files of the BM experiments for this data set.
    :param compute_test_loss: Whether to compute the test loss (retraining required) or not.
    :param new_use_case: Whether the cut / time budget defines a new use case or not.
    :return: tb_loss_df
    """

    trial_id_list = []
    algo_list = []
    hpo_tech_list = []
    worker_list = []
    warm_start_list = []
    budget_list = []
    tb_testloss_list = []
    tb_valloss_list = []
    fast_tech_list = []

    data_is_loaded = False

    # Setup variants (# workers, warm start (Yes / No))
    setup_vars = [(1, False), (8, False), (1, True)]

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

                val_loss_list = []
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

                    # Append min validation loss within the time budget
                    val_loss_list.append(run_df.loc[min_val_loss_idx, 'val_losses'])

                    if compute_test_loss:

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

                if compute_test_loss:
                    # Calculate the mean test loss over all runs of this HPO-technique (within time budget)
                    test_loss_arr = np.array(test_loss_list)
                    mean_test_loss = np.nanmean(test_loss_arr)

                # Calculate the mean validation loss over all runs of this HPO-technique (within time budget)
                mean_val_loss = np.nanmean(np.array(val_loss_list))

                if not new_use_case:
                    # Use the existing ID
                    trial_id_list.append(hpo_df['Trial-ID'].unique()[0])
                else:
                    # Create a new ID for this trial
                    trial_id_list.append(str(uuid.uuid4()))

                # Append
                algo_list.append(this_algo)
                hpo_tech_list.append(this_tech)
                worker_list.append(this_setup[0])
                warm_start_list.append(this_setup[1])
                budget_list.append(this_budget)
                tb_valloss_list.append(mean_val_loss)
                fast_tech_list.append(this_fastest_tech)

                if compute_test_loss:
                    tb_testloss_list.append(mean_test_loss)

    # Create a pd.DataFrame to store the results

    tb_loss_dict = {
        'Trial-ID': trial_id_list,
        'ML-algorithm': algo_list,
        'HPO-method': hpo_tech_list,
        'Workers': worker_list,
        'Warm start': warm_start_list,
        'Time Budget [s]': budget_list,
        'Min. validatino loss in time budget': tb_valloss_list,
        'Fastest HPO-technique': fast_tech_list
    }
    if compute_test_loss:
        tb_loss_dict['Min. test loss in time budget'] = tb_testloss_list

    tb_loss_df = pd.DataFrame(tb_loss_dict)

    return tb_loss_df


def compute_auc_for_time_budget(time_budget_df: pd.DataFrame, log_df: pd.DataFrame):
    """
    Compute the AUC of a learning curve for a given time budget.
    :return:
    """

    trial_id_list = []
    algo_list = []
    hpo_tech_list = []
    worker_list = []
    warm_start_list = []
    budget_list = []
    mean_auc_list = []
    fast_tech_list = []

    # Setup variants (# workers, warm start (Yes / No))
    setup_vars = [(1, False), (8, False), (1, True)]

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

            auc_list = []

            # Iterate over HPO-techniques
            for this_tech in hpo_techs:

                # Filter by HPO-technique
                hpo_df = use_case_df.loc[use_case_df['HPO-method'] == this_tech, :]

                # Run IDs
                runs = list(hpo_df['Run-ID'].unique())

                print('-----------------------------------------------------')
                print('Computing AUC for ' + this_tech + ' on ' + this_algo)

                # Iterate over runs per HPO-technique
                for this_run in runs:

                    # Filter by run
                    run_df = hpo_df.loc[hpo_df['Run-ID'] == this_run]

                    run_successful = run_df['run_successful'].to_numpy()[0]

                    if not run_successful:
                        continue

                    # Filter for time budget
                    sub_run_df = run_df.loc[run_df['timestamps'] <= this_budget, :]

                    min_loss = np.inf

                    trace_desc = []
                    i = 0

                    for idx, row in sub_run_df.iterrows():

                        this_loss = row['val_losses']
                        if i == 0:
                            min_loss = this_loss

                        elif this_loss < min_loss:
                            min_loss = this_loss

                        i += 1
                        trace_desc.append(min_loss)

                    auc_list.append(hpo_metrics.area_under_curve(trace_desc, lower_bound=0.0))

                mean_auc = np.nanmean(np.array(auc_list))

                trial_id_list.append(hpo_df['Trial-ID'].unique()[0])
                algo_list.append(this_algo)
                hpo_tech_list.append(this_tech)
                worker_list.append(this_setup[0])
                warm_start_list.append(this_setup[1])
                budget_list.append(this_budget)
                mean_auc_list.append(mean_auc)
                fast_tech_list.append(this_fastest_tech)

    tb_auc_df = pd.DataFrame({
        'Trial-ID': trial_id_list,
        'ML-algorithm': algo_list,
        'HPO-method': hpo_tech_list,
        'Workers': worker_list,
        'Warm start': warm_start_list,
        'Time Budget [s]': budget_list,
        'AUC within Time Budget': mean_auc_list,
        'Fastest HPO-technique': fast_tech_list
    })

    return tb_auc_df


if __name__ == '__main__':

    # Identify time budgets
    # Find validation and test losses for a given time budget (computed or user defined)
    # Find AUC for a given time budget (computed or ?user defined?)
    # Include new information to metrics.csv file -> can be validation and test loss or AUC

    dataset = 'turbofan'

    # Flags
    identify_time_budgets = False

    get_losses_for_budget = True
    user_defined = True
    compute_test_loss = False

    compute_auc = False

    append_results_to_metrics = False

    # Read the aggregated log file -> pd.DataFrame
    log_path = './hpo_framework/results/' + dataset + '/logs_' + dataset + '.csv'
    log_df = pd.read_csv(log_path, index_col=0)

    # Find time budgets?
    if identify_time_budgets:

        # Identify time budgets for each use case -> minimum wall clock time (fastest optimization run)
        time_budget_df = find_time_budget(log_df)

        # Write results to .csv-file
        tb_path = './hpo_framework/results/' + dataset + '/time_budgets_' + dataset + '.csv'
        time_budget_df.to_csv(tb_path)

    # Compute loss values for time budgets?
    if get_losses_for_budget:

        # Compute the losses up to the maximum time budget (defined by the fastest optimization run)
        if not user_defined:
            tb_path = './hpo_framework/results/' + dataset + '/time_budgets_' + dataset + '.csv'
            computed_budget_df = pd.read_csv(tb_path, index_col=0)
            tb_loss_df = get_losses_in_time_budget(computed_budget_df, log_df, compute_test_loss=compute_test_loss,
                                                   new_use_case=False)
            filestr = 'losses_for_max_time_budget_'

        # Compute the losses up to the user defined time budget (can be an point prior to the maximum time budget)
        else:
            tb_path = './hpo_framework/results/turbofan/my_cuts_turbofan.csv'
            user_budget_df = pd.read_csv(tb_path, index_col=0)
            tb_loss_df = get_losses_in_time_budget(user_budget_df, log_df, compute_test_loss=compute_test_loss,
                                                   new_use_case=True)
            filestr = 'losses_for_user_def_budget_'

        # Write results to .csv-file
        tb_loss_path = './hpo_framework/results/' + dataset + '/' + filestr + dataset + '.csv'
        tb_loss_df.to_csv(tb_loss_path)

    # Compute AUC metric?
    if compute_auc:

        # Compute the AUC of the learning curves up to the maximum time budget
        tb_path = './hpo_framework/results/' + dataset + '/time_budgets_' + dataset + '.csv'
        computed_budget_df = pd.read_csv(tb_path, index_col=0)
        tb_auc_df = compute_auc_for_time_budget(computed_budget_df, log_df)

        # Write results to .csv-file
        tb_auc_path = './hpo_framework/results/' + dataset + '/auc_in_max_time_budget_' + dataset + '.csv'
        tb_auc_df.to_csv(tb_auc_path)

    # # TODO: Modify to add AUC, validaton loss, ...
    # write_test_losses_to_metrics = False
    # if write_test_losses_to_metrics:
    #     # PART 3: Write the results to the metrics file
    #     metrics_path = './hpo_framework/results/' + dataset + '/metrics_' + dataset + '.csv'
    #     tb_loss_path = './hpo_framework/results/' + dataset + '/min_losses_in_time_budget_' + dataset + '.csv'
    #
    #     # Load .csv-files
    #     metrics_df = pd.read_csv(metrics_path, index_col=0)
    #     tb_df = pd.read_csv(tb_loss_path, index_col=0)
    #
    #     # Count the number of modified lines in the metrics file
    #     col_count = 0
    #
    #     # Iterate through the rows in the tb_df
    #     for idx, row in tb_df.iterrows():
    #         # Query results
    #         trial_id = row['Trial-ID']
    #         min_test_loss = row['Min. test loss in time budget']
    #         time_budget = row['Time Budget [s]']
    #         fast_tech = row['Fastest HPO-technique']
    #
    #         # TODO: Check the behavior for new / unknown Trial-ID's
    #         # Write results to metrics_df
    #         metrics_df.loc[metrics_df['Trial-ID'] == trial_id, 'Time Budget [s]'] = time_budget
    #         metrics_df.loc[metrics_df['Trial-ID'] == trial_id, 'Min. avg. test loss in time budget'] = min_test_loss
    #         metrics_df.loc[metrics_df['Trial-ID'] == trial_id, 'Fastest HPO-Technique'] = fast_tech
    #
    #         print(str(col_count) + ' - Modified Trial: ' + trial_id)
    #         col_count += 1
    #
    #     print(str(col_count) + ' of ' + str(len(metrics_df)) + ' total lines have been modified!')
    #
    #     # Save modified metrics .csv-file
    #     metrics_df.to_csv(metrics_path)
