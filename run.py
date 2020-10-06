import time
import os
from pathlib import Path
import argparse

from hpo.hp_spaces import space_keras, space_rf, space_svr, space_xgb, space_ada, space_dt

from hpo.hpo_metrics import root_mean_squared_error
import preprocessing as pp
from hpo.trial import Trial

# Flag for debug mode (yes/no)
# yes -> set parameters for this trial in source code (below)
# no -> call script via terminal and pass arguments via argparse
debug = True

# Loading data and preprocessing
# >>> Linux OS and Windows require different path representations -> use pathlib <<<
abs_folder_path = os.path.abspath(path='datasets')
data_folder = Path(abs_folder_path)
train_file = "train.csv"
test_file = "test.csv"
submission_file = "sample_submission.csv"

train_raw = pp.load_data(data_folder, train_file)
test_raw = pp.load_data(data_folder, test_file)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)

if debug:
    # Set parameters manually
    hp_space = space_rf
    ml_algo = 'RandomForestRegressor'
    opt_schedule = [('optuna', 'RandomSearch'), ('skopt', 'SMAC')]
    # Possible schedule combinations [('optuna', 'CMA-ES'), ('optuna', 'RandomSearch'),
    # ('skopt', 'SMAC'), ('skopt', 'GPBO'), ('hpbandster', 'BOHB'), ('hpbandster', 'Hyperband'), ('robo', 'Fabolas'),
    # ('robo', 'Bohamiann'), ('hyperopt', 'TPE')]
    n_runs = 3
    n_func_evals = 45
    n_workers = 1
    loss_metric = root_mean_squared_error


else:
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")

    parser.add_argument('ml_algorithm', help="Specify the machine learning algorithm.",
                        choices=['RandomForestRegressor', 'KerasRegressor', 'XGBoostRegressor', 'SVR',
                                 'AdaBoostRegressor', 'DecisionTreeRegressor'])
    parser.add_argument('hpo_methods', help='Specify the HPO-methods.', nargs='*',
                        choices=['CMA-ES', 'RandomSearch', 'SMAC', 'GPBO', 'TPE', 'BOHB', 'Hyperband', 'Fabolas',
                                 'Bohamiann'])
    parser.add_argument('--n_func_evals', type=int, help='Number of function evaluations in each run.', default=15)
    parser.add_argument('--n_runs', type=int, help='Number of runs for each HPO-method (varying random seeds).', default=5)
    parser.add_argument('--n_workers', type=int, help='Number of workers to be used for the optimization (parallelization)',
                        default=1)
    parser.add_argument('--loss_metric', type=str, help='Loss metric', default='root_mean_squared_error',
                        choices=['root_mean_squared_error'])

    args = parser.parse_args()

    # Settings for this trial
    ml_algo = args.ml_algorithm
    n_runs = args.n_runs
    # Optimization budget is limited by the number of function evaluations (should be dividable by 3 for BOHB and HB
    # to ensure comparability)
    n_func_evals = args.n_func_evals
    n_workers = args.n_workers

    # Create the optimization schedule by matching the hpo-methods with their libraries
    opt_schedule = []
    for this_method in args.hpo_methods:

        if this_method == 'SMAC' or this_method == 'GPBO':
            opt_schedule.append(('skopt', this_method))

        elif this_method == 'CMA-ES' or this_method == 'RandomSearch':
            opt_schedule.append(('optuna', this_method))

        elif this_method == 'TPE':
            opt_schedule.append(('hyperopt', this_method))

        elif this_method == 'BOHB' or this_method == 'Hyperband':
            opt_schedule.append(('hpbandster', this_method))

        elif this_method == 'Fabolas' or this_method == 'Bohamiann':
            opt_schedule.append(('robo', this_method))

        else:
            raise Exception('Something went wrong! Please check the for-loop that matches HPO-methods and libraries.')

    # Select the hyperparameter space according to the ML-algorithm
    if ml_algo == 'RandomForestRegressor':
        hp_space = space_rf

    elif ml_algo == 'KerasRegressor':
        hp_space = space_keras

    elif ml_algo == 'XGBoostRegressor':
        hp_space = space_xgb

    elif ml_algo == 'SVR':
        hp_space = space_svr

    elif ml_algo == 'AdaBoostRegressor':
        hp_space = space_ada

    elif ml_algo == 'DecisionTreeRegrssor':
        hp_space = space_dt

    else:
        raise Exception('For this ML-algorithm no hyperparameter space has been defined yet.')

    # Identify the correct loss loss_metric
    if args.loss_metric == 'root_mean_squared_error':
        loss_metric = root_mean_squared_error

    else:
        raise Exception('This loss metric has not yet been implemented.')


# Display a summary of the trial settings
print('Optimize: ' + ml_algo)
print('With HPO-methods: ')
for this_tuple in opt_schedule:
    print(this_tuple[1])
print('------')
print('Setup: ' + str(n_func_evals) + ' evaluations, ' + str(n_runs) + ' runs, ' + str(n_workers) + ' worker(s).')
print('------')
print('Optimization schedule: ', opt_schedule)


# Create a new trial
trial = Trial(hp_space=hp_space, ml_algorithm=ml_algo, optimization_schedule=opt_schedule,
              metric=loss_metric, n_runs=n_runs, n_func_evals=n_func_evals, n_workers=n_workers,
              x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val)

# Run the optimizations
res = trial.run()

# Analyze the results
print('Best configuration found:')
print(trial.get_best_trial_result(res))

abs_results_path = os.path.abspath(path='hpo/results')
res_folder = Path(abs_results_path)

curves = trial.plot_learning_curve(res)
# curves.show()

curves_str = 'learning_curves_' + ml_algo + '_' + str(time.strftime("%Y_%m_%d %H-%M-%S", time.gmtime())) + '.jpg'
curves_path = os.path.join(res_folder, curves_str)
curves.savefig(fname=curves_path)

metrics, metrics_df = trial.get_metrics(res)

metrics_str = 'metrics_' + ml_algo + '_' + str(time.strftime("%Y_%m_%d %H-%M-%S", time.gmtime())) + '.csv'
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
