import skopt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestRegressor

from hpo.skopt_optimizer import SkoptOptimizer
from hpo.optuna_optimizer import OptunaOptimizer
from hpo.hpo_metrics import rmse
import preprocessing as pp
from hpo.trial import Trial

# Loading data and preprocessing
FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)

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

# Initialize a Trial-object and run the optimization
ML_AlGO = 'XGBoostRegressor'
# HPO_LIB = 'optuna'
# HPO_METHOD = 'TPE'
N_RUNS = 2
BUDGET = 10
N_WORKERS = 1
# OPT_Schedule = [('optuna', 'TPE'), ('skopt', 'SMAC')]
OPT_Schedule = [('optuna', 'TPE')]

trial = Trial(hp_space=space_xgb, ml_algorithm=ML_AlGO, optimization_schedule=OPT_Schedule, metric=rmse,
              n_runs=N_RUNS, budget=BUDGET, n_workers=N_WORKERS,
              x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val)

res = trial.run()

trial.plot_learning_curve(res)

print(trial.get_best_trial_result(res))

###########################################
'''
# RF_Optimizer = OptunaOptimizer(hp_space=space_rf, hpo_method='TPE', ml_algorithm='RandomForestRegressor',
#                                x_train=X_train, x_val=X_val, y_train=y_train, y_val=y_val, metric=rmse,
#                                budget=10)

# RF_Optimizer = SkoptOptimizer(hp_space=space_rf, hpo_method='SMAC', ml_algorithm='RandomForestRegressor',
#                               x_train=X_train, x_val=X_val, y_train=y_train, y_val=y_val, metric=rmse,
#                               budget=10)

# res = RF_Optimizer.optimize()
# 
# best_config = RF_Optimizer.get_best_configuration(res)
# 
# print(best_config)
# 
# RF_Optimizer.plot_learning_curve(res)
'''

'''
# SKOPT_OPTIMIZER
# RF_Optimizer.plot_learning_curve(res)

best_config = RF_Optimizer.get_best_configuration((res))

print(best_config)

reg = RandomForestRegressor(random_state=0, **best_config)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_val)
score = sqrt(mean_squared_error(y_val, y_pred))

print('Optimization score: ', RF_Optimizer.get_best_score(res))
print('Validation score: ', score)
'''
