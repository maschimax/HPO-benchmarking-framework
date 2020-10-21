import numpy as np
from robo.fmin import bayesian_optimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import skopt
import time

from datasets.dummy import preprocessing as pp

FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)

# HP-space
space_rf = [skopt.space.Integer(1, 200, name='n_estimators'),
            skopt.space.Integer(1, 80, name='max_depth'),
            skopt.space.Integer(1, 30, name='min_samples_leaf'),
            skopt.space.Integer(2, 20, name='min_samples_split'),
            skopt.space.Categorical(['auto', 'sqrt'], name='max_features')]

lower = np.array([1, 1, 1, 2])
upper = np.array([200, 80, 30, 20])
hps = ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split']


def train_evaluate_rf(X_train, y_train, X_val, y_val, params):
    model = RandomForestRegressor(**params, random_state=0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    val_loss = sqrt(mean_squared_error(y_val, y_pred))

    return val_loss


def objective_func_bo(x):  # objective function to be minimized
    params = {}
    for i in range(len(x)):
        params[hps[i]] = int(x[i])

    start_time = time.time()

    val_loss = train_evaluate_rf(X_train, y_train, X_val, y_val, params)

    finish_time = time.time() - start_time

    print('Loss: % f, Time: %f ' % (val_loss, finish_time))

    return val_loss


def objective_func_fabolas(x, s):
    params = {}
    for i in range(len(x)):
        params[hps[i]] = int(x[i])

    # n_train = len(X_train)
    n_budget = s
    idx_train = np.random.randint(low=0, high=n_budget, size=n_budget)
    x_train_fab = X_train.iloc[idx_train]
    y_train_fab = y_train.iloc[idx_train]

    start_time = time.time()

    val_loss = train_evaluate_rf(x_train_fab, y_train_fab, X_val, y_val, params)

    cost = time.time() - start_time

    print('loss: % f, Cost: % f ' % (val_loss, cost))

    return val_loss, cost


# results = bayesian_optimization(objective_func_bo, lower=lower, upper=upper, num_iterations=5)
# results = fabolas(objective_func_fabolas, lower=lower, upper=upper, s_min=100, s_max=50000, num_iterations=120)
results = bayesian_optimization(objective_func_bo, lower, upper, model_type='bohamiann', num_iterations=20)


bla = 0
