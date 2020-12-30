import time
import os
from pathlib import Path
import argparse
import skopt
import warnings
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

from hpo_framework.hp_spaces import space_keras, space_rf_reg, space_rf_clf, \
    space_svm, space_xgb, space_ada_reg, space_ada_clf, space_dt, \
    space_linr, space_knn, space_lgb, space_logr, space_nb, \
    space_mlp, space_elnet
from hpo_framework.hpo_metrics import root_mean_squared_error, f1_loss, accuracy_loss, rul_loss_score
from hpo_framework.trial import Trial
import datasets.dummy.preprocessing as pp
from datasets.Scania_APS_Failure.scania_preprocessing import scania_loading_and_preprocessing
from datasets.Turbofan_Engine_Degradation.turbofan_preprocessing import turbofan_loading_and_preprocessing
from datasets.Sensor_System_Production.sensor_loading_and_balancing import sensor_loading_and_preprocessing
from datasets.Blisk.blisk_preprocessing import blisk_loading_and_preprocessing
from datasets.Mining_Process.mining_preprocessing import mining_loading_and_preprocessing
from datasets.Faulty_Steel_Plates.steel_preprocessing import steel_loading_and_preprocessing

# Flag for debug mode (yes/no)
# yes (True) -> set parameters for this trial in source code (below)
# no (False) -> call script via terminal and pass arguments via argparse
debug = False

if debug:
    # Set parameters manually
    dataset = 'sensor'  # Flag for the ML use case / dataset to be used
    hp_space = space_keras
    ml_algo = 'KerasClassifier'
    opt_schedule = [('optuna', 'TPE')]
    # Possible schedule combinations [('optuna', 'CMA-ES'), ('optuna', 'RandomSearch'),
    # ('skopt', 'SMAC'), ('skopt', 'GPBO'), ('hpbandster', 'BOHB'), ('hpbandster', 'Hyperband'), ('robo', 'Fabolas'),
    # ('robo', 'Bohamiann'), ('optuna', 'TPE')]
    n_runs = 1
    n_func_evals = 200
    n_workers = 4
    loss_metric = f1_loss
    loss_metric_str = 'F1-loss'
    do_warmstart = 'No'
    plot_learning_curves = 'No'
    use_gpu = 'No'
    cross_validation = 'No'
    shuffle = 'Yes'

else:
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization Benchmarking Framework")

    parser.add_argument('ml_algorithm', help="Specify the machine learning algorithm.",
                        choices=['RandomForestRegressor', 'RandomForestClassifier', 'KerasRegressor', 'KerasClassifier',
                                 'XGBoostRegressor', 'XGBoostClassifier', 'SVR', 'SVC', 'AdaBoostRegressor',
                                 'AdaBoostClassifier',
                                 'DecisionTreeRegressor', 'DecisionTreeClassifier', 'LinearRegression',
                                 'LogisticRegression', 'KNNRegressor', 'KNNClassifier',
                                 'LGBMRegressor',
                                 'LGBMClassifier', 'NaiveBayes', 'MLPRegressor', 'MLPClassifier', 'ElasticNet'])

    parser.add_argument('hpo_methods', help='Specify the HPO-methods.', nargs='*',
                        choices=['CMA-ES', 'RandomSearch', 'SMAC', 'GPBO', 'TPE', 'BOHB', 'Hyperband', 'Fabolas',
                                 'Bohamiann'])

    parser.add_argument('--dataset', type=str, help='Dataset / use case.', default='scania',
                        choices=['scania', 'turbofan', 'mining', 'steel', 'sensor', 'blisk', 'dummy'])

    parser.add_argument('--n_func_evals', type=int, help='Number of function evaluations in each run.', default=15)

    parser.add_argument('--n_runs', type=int, help='Number of runs for each HPO-method (varying random seeds).',
                        default=5)

    parser.add_argument('--n_workers', type=int,
                        help='Number of workers to be used for the optimization (parallelization)',
                        default=1)

    parser.add_argument('--loss_metric', type=str, help='Loss metric', default='root_mean_squared_error',
                        choices=['root_mean_squared_error', 'F1-loss', 'Accuracy-loss', 'RUL-loss'])

    parser.add_argument('--warmstart', type=str,
                        help="Use the algorithm's default HP-configuration for warmstart (yes/no).",
                        default='No', choices=['Yes', 'No'])

    parser.add_argument('--plot_learning_curves', type=str, help='Show learning curves (yes/no).',
                        default='No', choices=['Yes', 'No'])

    parser.add_argument('--gpu', type=str, help='Use GPU resources if available (yes/no).', default='No',
                        choices=['Yes', 'No'])

    parser.add_argument('--cross_validation', type=str, help='Apply cross validation (yes/no).', default='No',
                        choices=['Yes', 'No'])

    parser.add_argument('--shuffle', type=str, help='Shuffle training data (yes/no)', default='Yes',
                        choices=['Yes', 'No'])

    args = parser.parse_args()

    # Settings for this trial
    ml_algo = args.ml_algorithm
    dataset = args.dataset
    n_runs = args.n_runs
    # Optimization budget is limited by the number of function evaluations (should be dividable by 3 for BOHB and HB
    # to ensure comparability)
    n_func_evals = args.n_func_evals
    n_workers = args.n_workers
    do_warmstart = args.warmstart
    plot_learning_curves = args.plot_learning_curves
    use_gpu = args.gpu
    cross_validation = args.cross_validation
    shuffle = args.shuffle

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

    elif ml_algo == 'KerasRegressor' or ml_algo == 'KerasClassifier':
        hp_space = space_keras

    elif ml_algo == 'XGBoostRegressor' or ml_algo == 'XGBoostClassifier':
        hp_space = space_xgb

    elif ml_algo == 'SVR' or ml_algo == 'SVC':
        hp_space = space_svm

    elif ml_algo == 'AdaBoostRegressor':
        hp_space = space_ada_reg

    elif ml_algo == 'AdaBoostClassifier':
        hp_space = space_ada_clf

    elif ml_algo == 'DecisionTreeRegressor' or ml_algo == 'DecisionTreeClassifier':
        hp_space = space_dt

    elif ml_algo == 'LinearRegression':
        hp_space = space_linr

    elif ml_algo == 'KNNRegressor' or ml_algo == 'KNNClassifier':
        hp_space = space_knn

    elif ml_algo == 'LGBMRegressor' or ml_algo == 'LGBMClassifier':
        hp_space = space_lgb

    elif ml_algo == 'LogisticRegression':
        hp_space = space_logr

    elif ml_algo == 'NaiveBayes':
        hp_space = space_nb

    elif ml_algo == 'MLPRegressor' or ml_algo == 'MLPClassifier':
        hp_space = space_mlp

    elif ml_algo == 'ElasticNet':
        hp_space = space_elnet

    else:
        raise Exception('For this ML-algorithm no hyperparameter space has been defined yet.')

    # Identify the correct loss metric
    loss_metric_str = args.loss_metric
    if loss_metric_str == 'root_mean_squared_error':
        loss_metric = root_mean_squared_error

    elif loss_metric_str == 'F1-loss':
        loss_metric = f1_loss

    elif loss_metric_str == 'Accuracy-loss':
        loss_metric = accuracy_loss

    elif loss_metric_str == 'RUL-loss':
        loss_metric = rul_loss_score

    else:
        raise Exception('This loss metric has not yet been implemented.')

# Loading and preprocessing of the selected data set
if dataset == 'dummy':
    # Loading data and preprocessing
    # >>> Linux OS and Windows require different path representations -> use pathlib <<<
    dataset_path = os.path.abspath(path='datasets/dummy')
    data_folder = Path(dataset_path)
    train_file = "train.csv"
    test_file = "test.csv"
    submission_file = "sample_submission.csv"

    train_raw = pp.load_data(data_folder, train_file)
    test_raw = pp.load_data(data_folder, test_file)

    X_train, y_train, X_test, y_test, _ = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                     count_encoding=False)

elif dataset == 'scania':

    X_train, X_test, y_train, y_test = scania_loading_and_preprocessing()

elif dataset == 'turbofan':

    X_train, X_test, y_train, y_test = turbofan_loading_and_preprocessing()

elif dataset == 'sensor':

    X_train, X_test, y_train, y_test = sensor_loading_and_preprocessing()

    # For Keras models, apply one-hot-encoding (multiclass-classification problem)
    if ml_algo == 'KerasClassifier' or ml_algo == 'KerasRegressor':

        # Apply scikit-learn's OneHotEncoder
        oh_enc = OneHotEncoder(sparse=False, handle_unknown='error')
        y_train_oh = oh_enc.fit_transform(np.array(y_train).reshape(-1, 1))
        y_test_oh = oh_enc.transform(np.array(y_test).reshape(-1, 1))

        # Transform numpy arrays to pandas DataFrames
        y_train = pd.DataFrame(y_train_oh)
        y_test = pd.DataFrame(y_test_oh)

elif dataset == 'blisk':

    X_train, X_test, y_train, y_test = blisk_loading_and_preprocessing()

elif dataset == 'mining':

    X_train, X_test, y_train, y_test = mining_loading_and_preprocessing()

elif dataset == 'steel':

    X_train, X_test, y_train, y_test = steel_loading_and_preprocessing()

    if ml_algo == 'RandomForestClassifier' or ml_algo == 'SVC' or \
            ml_algo == 'LogisticRegression' or ml_algo == 'NaiveBayes' or \
            ml_algo == 'DecisionTreeClassifier' or ml_algo == 'KNNClassifier' or \
            ml_algo == 'AdaBoostClassifier' or ml_algo == 'MLPClassifier' or \
            ml_algo == 'XGBoostClassifier' or ml_algo == 'LGBMClassifier':

        # Reverse one hot encoding // sklearn, xgboost and lightgbm models require a label encoded label vector
        # for multiclass-classification
        for y in [y_train, y_test]:
            for iter_tuple in y.itertuples():
                idx = iter_tuple.Index
                fault_id = [i for i, j in enumerate(iter_tuple[1:]) if j > 0.5]
                y.loc[idx, 'Fault_ID'] = fault_id

        y_train = y_train.loc[:, 'Fault_ID']
        y_test = y_test.loc[:, 'Fault_ID']

else:
    raise Exception('Unknown dataset / use-case.')

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

if n_func_evals <= 10:
    warnings.warn('Some HPO-methods expect a budget of at least 20 evaluations. The optimization might fail.')

if use_gpu == 'No':
    gpu = False
else:
    gpu = True

if cross_validation == 'No':
    cv = False
else:
    cv = True

if shuffle == 'Yes':
    shuffle_data = True
else:
    shuffle_data = False

# Create a new trial
trial = Trial(hp_space=hp_space, ml_algorithm=ml_algo, optimization_schedule=opt_schedule,
              metric=loss_metric, n_runs=n_runs, n_func_evals=n_func_evals, n_workers=n_workers,
              x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test, do_warmstart=do_warmstart,
              gpu=gpu, cross_val=cv, shuffle=shuffle_data)

# Run the optimization
res = trial.run()

abs_results_path = os.path.abspath(path='hpo_framework/results')

res_folder = os.path.join(abs_results_path, dataset)
if not os.path.isdir(res_folder):
    os.makedirs(res_folder, exist_ok=True)

log_folder = os.path.join(res_folder, 'logs')
if not os.path.isdir(log_folder):
    os.makedirs(log_folder, exist_ok=True)

time_str = str(time.strftime("%Y_%m_%d %H-%M-%S", time.localtime()))

# Analyze the results
print('Best configuration found:')
print(trial.get_best_trial_result(res))

# Determine the number of HPs for each HP type (continuous, integer-valued, categorical)
num_params = {'continuous': 0, 'integer': 0, 'categorical': 0}
for param in hp_space:

    if type(param) == skopt.space.space.Real:
        num_params['continuous'] = num_params['continuous'] + 1

    elif type(param) == skopt.space.space.Integer:
        num_params['integer'] = num_params['integer'] + 1

    elif type(param) == skopt.space.space.Categorical:
        num_params['categorical'] = num_params['categorical'] + 1

    else:
        continue

if gpu:
    gpu_str = 'GPU'
else:
    gpu_str = 'CPU'

if do_warmstart == 'Yes':
    warmstart_str = 'Warmstart'
else:
    warmstart_str = 'NoWarmstart'

# Optimization results
for opt_tuple in res.keys():
    res_df = res[opt_tuple].trial_result_df
    res_df['dataset'] = dataset
    res_df['# cont. HPs'] = num_params['continuous']
    res_df['# int. HPs'] = num_params['integer']
    res_df['# cat. HPs'] = num_params['categorical']
    res_df['loss_metric'] = loss_metric_str

    res_str_csv = dataset + '_' + ml_algo + '_' + opt_tuple[1] + '_' + str(n_workers) + 'Workers' + '_' + gpu_str + \
        '_' + warmstart_str + '_' + time_str + '.csv'
    res_path_csv = os.path.join(log_folder, res_str_csv)

    # res_str_json = use_case + '_' + ml_algo + '_' + opt_tuple[1] + '_' + time_str + '.json'
    # res_path_json = os.path.join(log_folder, res_str_json)

    # Don't reset index inplace!
    this_res_df = res_df.reset_index(drop=True, inplace=False)
    this_res_df.to_csv(res_path_csv)
    # res_df.to_json(res_path_json)

# Learning curves
if plot_learning_curves == 'Yes':
    curves = trial.plot_learning_curve(res)
    curves_str = 'learning_curves_' + dataset + '_' + ml_algo + '_' + time_str + '.jpg'
    curves_path = os.path.join(res_folder, curves_str)
    curves.savefig(fname=curves_path)

# Metrics
metrics, metrics_df = trial.get_metrics(res)
metrics_df['dataset'] = dataset
metrics_df['# cont. HPs'] = num_params['continuous']
metrics_df['# int. HPs'] = num_params['integer']
metrics_df['# cat. HPs'] = num_params['categorical']
metrics_df['loss_metric'] = loss_metric_str

hpo_str = '_'
for opt_tuple in opt_schedule:
    hpo_str = hpo_str + opt_tuple[1]

metrics_str = 'metrics_' + dataset + '_' + ml_algo + hpo_str + '_' + str(n_workers) + 'Workers' + '_' \
              + gpu_str + '_' + warmstart_str + '_' + time_str + '.csv'
metrics_path = os.path.join(res_folder, metrics_str)
metrics_df.to_csv(metrics_path)

# Hyperparameter space
space_plots = trial.plot_hp_space(res)
for opt_tuple in space_plots.keys():
    this_plot = space_plots[opt_tuple]
    this_hpo_method = opt_tuple[1]
    space_str = 'hp_space_' + dataset + '_' + ml_algo + '_' + this_hpo_method + '_' + str(n_workers) + 'Workers' + '_'\
                + gpu_str + '_' + warmstart_str + '_' + time_str + '.svg'
    space_path = os.path.join(res_folder, space_str)
    this_plot.savefig(fname=space_path)
