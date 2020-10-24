import time
import os
from pathlib import Path
import argparse

from hpo_framework.hp_spaces import space_keras, space_rf_reg, space_rf_clf, space_svr, space_svc, space_xgb,\
    space_ada, space_dt, space_linr, space_knn_reg, space_lgb_clf
from hpo_framework.hpo_metrics import root_mean_squared_error, f1_loss, accuracy_loss
from hpo_framework.trial import Trial
import datasets.dummy.preprocessing as pp
from datasets.Scania_APS_Failure.scania_preprocessing import scania_loading_and_preprocessing

# Flag for the ML use case / dataset to be used
use_case = 'scania'

if use_case == 'dummy':
    # Loading data and preprocessing
    # >>> Linux OS and Windows require different path representations -> use pathlib <<<
    abs_folder_path = os.path.abspath(path='datasets/dummy')
    data_folder = Path(abs_folder_path)
    train_file = "train.csv"
    test_file = "test.csv"
    submission_file = "sample_submission.csv"

    train_raw = pp.load_data(data_folder, train_file)
    test_raw = pp.load_data(data_folder, test_file)

    X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                        count_encoding=False)

elif use_case == 'scania':

    X_train, X_val, y_train, y_val = scania_loading_and_preprocessing()

else:
    raise Exception('Unknown dataset / use-case.')

# Flag for debug mode (yes/no)
# yes (True) -> set parameters for this trial in source code (below)
# no (False) -> call script via terminal and pass arguments via argparse
debug = True

if debug:
    # Set parameters manually
    hp_space = space_lgb_clf
    ml_algo = 'LGBMClassifier'
    opt_schedule = [('optuna', 'TPE')]
    # Possible schedule combinations [('optuna', 'CMA-ES'), ('optuna', 'RandomSearch'),
    # ('skopt', 'SMAC'), ('skopt', 'GPBO'), ('hpbandster', 'BOHB'), ('hpbandster', 'Hyperband'), ('robo', 'Fabolas'),
    # ('robo', 'Bohamiann'), ('optuna', 'TPE')]
    n_runs = 1
    n_func_evals = 20
    n_workers = 4
    loss_metric = accuracy_loss
    do_warmstart = 'No'

else:
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")

    parser.add_argument('ml_algorithm', help="Specify the machine learning algorithm.",
                        choices=['RandomForestRegressor', 'RandomForestClassifier', 'KerasRegressor',
                                 'XGBoostRegressor', 'SVR', 'SVC', 'AdaBoostRegressor', 'DecisionTreeRegressor',
                                 'LinearRegression', 'KNNRegressor', 'LGBMClassifier'])
    parser.add_argument('hpo_methods', help='Specify the HPO-methods.', nargs='*',
                        choices=['CMA-ES', 'RandomSearch', 'SMAC', 'GPBO', 'TPE', 'BOHB', 'Hyperband', 'Fabolas',
                                 'Bohamiann'])
    parser.add_argument('--n_func_evals', type=int, help='Number of function evaluations in each run.', default=15)
    parser.add_argument('--n_runs', type=int, help='Number of runs for each HPO-method (varying random seeds).',
                        default=5)
    parser.add_argument('--n_workers', type=int,
                        help='Number of workers to be used for the optimization (parallelization)',
                        default=1)
    parser.add_argument('--loss_metric', type=str, help='Loss metric', default='root_mean_squared_error',
                        choices=['root_mean_squared_error', 'F1-loss'])
    parser.add_argument('--warmstart', type=str,
                        help="Use the algorithm's default HP-configuration for warmstart (yes/no).",
                        default='No', choices=['Yes', 'No'])

    args = parser.parse_args()

    # Settings for this trial
    ml_algo = args.ml_algorithm
    n_runs = args.n_runs
    # Optimization budget is limited by the number of function evaluations (should be dividable by 3 for BOHB and HB
    # to ensure comparability)
    n_func_evals = args.n_func_evals
    n_workers = args.n_workers
    do_warmstart = args.warmstart

    # Create the optimization schedule by matching the hpo-methods with their libraries
    opt_schedule = []
    for this_method in args.hpo_methods:

        if this_method == 'SMAC' or this_method == 'GPBO':
            opt_schedule.append(('skopt', this_method))

        elif this_method == 'CMA-ES' or this_method == 'RandomSearch' or this_method == 'TPE':
            opt_schedule.append(('optuna', this_method))

        # elif this_method == 'TPE':
        #     opt_schedule.append(('hyperopt', this_method))

        elif this_method == 'BOHB' or this_method == 'Hyperband':
            opt_schedule.append(('hpbandster', this_method))

        elif this_method == 'Fabolas' or this_method == 'Bohamiann':
            opt_schedule.append(('robo', this_method))

        else:
            raise Exception('Something went wrong! Please check the for-loop that matches HPO-methods and libraries.')

    # Select the hyperparameter space according to the ML-algorithm
    if ml_algo == 'RandomForestRegressor':
        hp_space = space_rf_reg

    elif ml_algo == 'RandomForestClassifier':
        hp_space = space_rf_clf

    elif ml_algo == 'KerasRegressor':
        hp_space = space_keras

    elif ml_algo == 'XGBoostRegressor':
        hp_space = space_xgb

    elif ml_algo == 'SVR':
        hp_space = space_svr

    elif ml_algo == 'SVC':
        hp_space = space_svc

    elif ml_algo == 'AdaBoostRegressor':
        hp_space = space_ada

    elif ml_algo == 'DecisionTreeRegrssor':
        hp_space = space_dt

    elif ml_algo == 'LinearRegression':
        hp_space = space_linr

    elif ml_algo == 'KNNRegressor':
        hp_space = space_knn_reg

    elif ml_algo == 'LGBClassifier':
        hp_space = space_lgb_clf

    else:
        raise Exception('For this ML-algorithm no hyperparameter space has been defined yet.')

    # Identify the correct loss metric
    if args.loss_metric == 'root_mean_squared_error':
        loss_metric = root_mean_squared_error

    elif args.loss_metric == 'F1-loss':
        loss_metric = f1_loss

    else:
        raise Exception('This loss metric has not yet been implemented.')

# Display a summary of the trial settings
print('Optimize: ' + ml_algo)
print('With HPO-methods: ')
for this_tuple in opt_schedule:
    print(this_tuple[1])
print('------')
print('Setup: ' + str(n_func_evals) + ' evaluations, ' + str(n_runs) + ' runs, ' + str(n_workers) +
      ' worker(s), warmstart: ' + do_warmstart + '.')
print('------')
print('Optimization schedule: ', opt_schedule)

# Create a new trial
trial = Trial(hp_space=hp_space, ml_algorithm=ml_algo, optimization_schedule=opt_schedule,
              metric=loss_metric, n_runs=n_runs, n_func_evals=n_func_evals, n_workers=n_workers,
              x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, do_warmstart=do_warmstart)

# Run the optimizations
res = trial.run()

abs_results_path = os.path.abspath(path='hpo_framework/results')
res_folder = Path(abs_results_path)
abs_log_path = os.path.abspath(path='hpo_framework/results/logs')
log_folder = Path(abs_log_path)

time_str = str(time.strftime("%Y_%m_%d %H-%M-%S", time.gmtime()))

# Analyze the results
print('Best configuration found:')
print(trial.get_best_trial_result(res))

# Optimization results
for opt_tuple in res.keys():
    res_df = res[opt_tuple].trial_result_df

    res_str_csv = ml_algo + '_' + opt_tuple[1] + '_' + time_str + '.csv'
    res_path_csv = os.path.join(log_folder, res_str_csv)

    # res_str_json = ml_algo + '_' + opt_tuple[1] + '_' + time_str + '.json'
    # res_path_json = os.path.join(log_folder, res_str_json)

    res_df.reset_index(drop=True, inplace=True)
    res_df.to_csv(res_path_csv)
    # res_df.to_json(res_path_json)

# Learning curves
curves = trial.plot_learning_curve(res)
curves_str = 'learning_curves_' + ml_algo + '_' + time_str + '.jpg'
curves_path = os.path.join(res_folder, curves_str)
curves.savefig(fname=curves_path)

# Hyperparameter space
space_plots = trial.plot_hp_space(res)
for opt_tuple in space_plots.keys():
    this_plot = space_plots[opt_tuple]
    this_hpo_method = opt_tuple[1]
    space_str = 'hp_space_' + ml_algo + '_' + this_hpo_method + '_' + time_str + '.jpg'
    space_path = os.path.join(res_folder, space_str)
    this_plot.savefig(fname=space_path)

# Metrics
metrics, metrics_df = trial.get_metrics(res)
metrics_str = 'metrics_' + ml_algo + '_' + time_str + '.csv'
metrics_path = os.path.join(res_folder, metrics_str)
metrics_df.to_csv(metrics_path)

bla = 0

# # Train best model on the whole data set
# x_data = pd.concat(objs=[X_train, X_val], axis=0)
# y_data = pd.concat(objs=[y_train, y_val], axis=0)
#
# best_trial = trial.get_best_trial_result(res)
#
# best_model = XGBRegressor(**best_trial['HP-configuration'])
# best_model.fit(x_data, y_data)
#
# y_pred = best_model.predict(X_test)
#
# sample_submission = pp.load_data(folder=dir_name, file=submission_file)
# this_submission = sample_submission.copy()
#
# this_submission['SalePrice'] = y_pred
# this_submission.to_csv(r'C:\Users\Max\Documents\GitHub\housing_regression\datasets\submission.csv')
