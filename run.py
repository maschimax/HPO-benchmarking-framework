import skopt
import time
import os
from pathlib import Path
import argparse

from hpo.hpo_metrics import root_mean_squared_error
import preprocessing as pp
from hpo.trial import Trial

parser = argparse.ArgumentParser(description="Hyperparameter Optimization")

parser.add_argument('ml_algorithm', help="Specify the machine learning algorithm.",
                    choices=['RandomForestRegressor', 'KerasRegressor', 'XGBoostRegressor', 'SVR'])
parser.add_argument('hpo_methods', help='Specify the HPO-methods.', nargs='*',
                    choices=['CMA-ES', 'RandomSearch', 'SMAC', 'GPBO', 'TPE', 'BOHB', 'Hyperband', 'Fabolas',
                             'Bohamiann'])
parser.add_argument('--n_func_evals', type=int, help='Number of function evaluations in each run.', default=15)
parser.add_argument('--n_runs', type=int, help='Number of runs for each HPO-method (varying random seeds).', default=5)
parser.add_argument('--n_workers', type=int, help='Number of workers to be used for the optimization (parallelization)',
                    default=1)

args = parser.parse_args()

print('Optimize: ' + args.ml_algorithm)
print('With HPO-methods: ')
for this_method in args.hpo_methods:
    print(this_method)
print('------')
print('Setup: ' + str(args.n_func_evals) + ' evaluations, ' + str(args.n_runs) + ' runs, ' +
      str(args.n_workers) + ' worker(s).')

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

# Define HP-space according to the skopt library
space_rf = [skopt.space.Integer(1, 200, name='n_estimators'),
            skopt.space.Integer(1, 80, name='max_depth'),
            skopt.space.Integer(1, 30, name='min_samples_leaf'),
            skopt.space.Integer(2, 20, name='min_samples_split'),
            skopt.space.Categorical(['auto', 'sqrt'], name='max_features')]
# skopt.space.Real(0.1, 0.9, name='max_samples')

space_svr = [skopt.space.Real(low=1e-3, high=1e+3, name='C'),
             skopt.space.Categorical(['scale', 'auto'], name='gamma'),
             skopt.space.Real(low=1e-3, high=1e+0, name='epsilon')]

space_keras = [skopt.space.Categorical([.0005, .001, .005, .01, .1], name='init_lr'),
               skopt.space.Categorical([8, 16, 32, 64], name='batch_size'),
               skopt.space.Categorical(['cosine', 'constant'], name='lr_schedule'),
               skopt.space.Categorical(['relu', 'tanh'], name='layer1_activation'),
               skopt.space.Categorical(['relu', 'tanh'], name='layer2_activation'),
               skopt.space.Categorical([16, 32, 64, 128, 256, 512], name='layer1_size'),
               skopt.space.Categorical([16, 32, 64, 128, 256, 512], name='layer2_size'),
               skopt.space.Categorical([.0, .3, .6], name='dropout1'),
               skopt.space.Categorical([.0, .3, .6], name='dropout2')]

space_xgb = [skopt.space.Categorical(['gbtree', 'gblinear', 'dart'], name='booster'),
             skopt.space.Integer(1, 200, name='n_estimators'),
             skopt.space.Integer(1, 80, name='max_depth')]

# Setting for the trial
ML_ALGO = args.ml_algorithm
N_RUNS = args.n_runs
# Optimization budget is limited by the number of function evaluations (should be dividable by 3 for BOHB and HB
# to ensure comparability)
N_FUNC_EVALS = args.n_func_evals
N_WORKERS = args.n_workers

# Create the optimization schedule by matching the hpo-methods with their libraries
opt_schedule = []
for this_method in args.hpo_methods:

    if this_method == 'SMAC' or this_method == 'GPBO':
        opt_schedule.append(('skopt', this_method))

    elif this_method == 'CMA-ES' or this_method == 'RandomSearch':
        opt_schedule.append(('optuna', this_method))

    elif this_method == 'TPE':
        opt_schedule.append(('hyperopt', this_method))

    elif this_method == 'BOHB' or this_method == 'Hyberband':
        opt_schedule.append(('hpbandster', this_method))

    elif this_method == 'Fabolas' or this_method == 'Bohamiann':
        opt_schedule.append(('robo', this_method))

    else:
        raise Exception('Something went wrong! Please check the for-loop that matches HPO-methods and libraries.')

# Select the hyperparameter space according to the ML-algorithm
if ML_ALGO == 'RandomForestRegressor':
    hp_space = space_rf

elif ML_ALGO == 'KerasRegressor':
    hp_space = space_keras

elif ML_ALGO == 'XGBoostRegressor':
    hp_space = space_xgb

elif ML_ALGO == 'SVR':
    hp_space = space_svr

else:
    raise Exception('For this ML-algorithm no hyperparameter space has been defined yet.')

# Possible schedule combinations -> OPT_Schedule = [('optuna', 'CMA-ES'), ('optuna', 'RandomSearch'),
# ('skopt', 'SMAC'), ('skopt', 'GPBO'), ('hpbandster', 'BOHB'), ('hpbandster', 'Hyperband'), ('robo', 'Fabolas'),
# ('robo', 'Bohamiann'), ('hyperopt', 'TPE')]
# OPT_Schedule = [('hyperopt', 'TPE'), ('skopt', 'SMAC'), ('hpbandster', 'BOHB'), ('optuna', 'RandomSearch')]

# Create a new trial
trial = Trial(hp_space=hp_space, ml_algorithm=ML_ALGO, optimization_schedule=opt_schedule, metric=root_mean_squared_error,
              n_runs=N_RUNS, n_func_evals=N_FUNC_EVALS, n_workers=N_WORKERS,
              x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val)

# Run the optimizations
res = trial.run()

# Analyze the results
print(trial.get_best_trial_result(res))

res_folder = r'C:\Users\Max\Documents\GitHub\housing_regression\hpo\results'

curves = trial.plot_learning_curve(res)
# curves.show()

curves_str = 'learning_curves_' + ML_ALGO + '_' + str(time.strftime("%Y_%m_%d %H-%M-%S", time.gmtime())) + '.jpg'
curves_path = os.path.join(res_folder, curves_str)
curves.savefig(fname=curves_path)

metrics, metrics_df = trial.get_metrics(res)

metrics_str = 'metrics_' + ML_ALGO + '_' + str(time.strftime("%Y_%m_%d %H-%M-%S", time.gmtime())) + '.csv'
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
