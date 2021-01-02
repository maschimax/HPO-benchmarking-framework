import pandas as pd
import numpy as np
import ast

from hpo_framework.baseoptimizer import BaseOptimizer
from datasets.Turbofan_Engine_Degradation.turbofan_preprocessing import turbofan_loading_and_preprocessing
from hpo_framework import hpo_metrics

dataset = 'turbofan'
log_path = './hpo_framework/results/' + dataset + '/logs_' + dataset + '.csv'

# Read the log file -> pd.DataFrame
log_df = pd.read_csv(log_path, index_col=0)

# Setup variants (# workers, warm start (Yes / No))
setup_vars = [(1, False), (8, False), (1, True)]

########################################################################################################################
# PART 1: Find time budgets for each use case (minimum wall clock time)

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

########################################################################################################################
# PART 2: Retrain the ML algorithms

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
                                         (time_budget_df['Warm start'] == this_setup[1]), 'Time budget [s]'].values[0]

        # Filter by ML algorithm -> use case
        use_case_df = setup_df.loc[setup_df['ML-algorithm'] == this_algo, :]

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

                # Index of minimum validation loss achieved within the time budget
                min_val_loss_idx = run_df.loc[run_df['timestamps'] <= this_budget, 'val_losses'].idxmin(axis=0)

                # Best found HP configuration within the time budget
                best_config_dict = ast.literal_eval(run_df.loc[min_val_loss_idx, 'configurations'])

                # Retrain the ML algorithm for this hyperparameter configuration and calculate the test loss
                loss_metric = run_df['loss_metric'].unique()[0]
                this_seed = run_df['random_seed'].unique()[0]

                if dataset == 'turbofan':

                    do_shuffle = True
                    X_train, X_test, y_train, y_test = turbofan_loading_and_preprocessing()

                else:
                    raise Exception('Please specify the data sets and the shuffling procedure for this data set!')

                if loss_metric == 'RUL-loss':

                    this_metric = hpo_metrics.rul_loss_score

                else:
                    raise Exception('Unknown loss metric!')

                dummy_optimizer = BaseOptimizer(hp_space=None, hpo_method=None, ml_algorithm=this_algo,
                                                x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test,
                                                metric=this_metric, n_func_evals=1,
                                                random_seed=this_seed, n_workers=this_setup[0], cross_val=False,
                                                shuffle=do_shuffle)

                test_loss = dummy_optimizer.train_evaluate_ml_model(params=best_config_dict, cv_mode=False,
                                                                    test_mode=True)

                # Save test loss values and best configs somewhere

                # Compute mean of test losses and save to metrics .csv file


# For each use case

    # Find the fastest HPO technique and determine the time budget

    # For each HPO technique

        # For each run

        # Find the best HP-configuration (based on the validation loss) until the time budget was consumed

        # Retrain the ML algorithm for this configuration

        # Write the time budget, the fastest HPO Technique, the validation loss and the test loss (both mean)
        # into the corresponding metrics file