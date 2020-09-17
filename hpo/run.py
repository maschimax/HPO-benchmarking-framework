import skopt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestRegressor

from hpo.skopt_optimizer import Skopt_optimizer
from hpo.optuna_optimizer import Optuna_optimizer
from hpo.hpo_metrics import rmse
import preprocessing as pp

FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)

space_rf = [skopt.space.Integer(1, 200, name='n_estimators'),
            skopt.space.Integer(1, 80, name='max_depth'),
            skopt.space.Integer(1, 30, name='min_samples_leaf'),
            skopt.space.Integer(2, 20, name='min_samples_split'),
            skopt.space.Categorical(['auto', 'sqrt'], name='max_features')]

RF_Optimizer = Optuna_optimizer(hp_space=space_rf, hpo_method='TPE', ml_algorithm='RandomForestRegressor',
                                x_train=X_train, x_val=X_val, y_train=y_train, y_val=y_val, metric=rmse,
                                budget=10)

res = RF_Optimizer.optimize()

best_config = RF_Optimizer.get_best_configuration((res))

RF_Optimizer.plot_learning_curve(res)

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
