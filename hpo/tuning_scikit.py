import skopt
from skopt.optimizer import gp_minimize
from skopt.optimizer import forest_minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

import preprocessing as pp

FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)


def train_evaluate_rf(X_train, y_train, X_val, y_val, params):
    rf_reg = RandomForestRegressor(**params, random_state=0)

    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_val)

    val_loss = sqrt(mean_squared_error(y_val, y_pred))

    return val_loss

def train_evaluate_keras(X_train, y_train, X_val, y_val, params):
    # Add function body here
    return val_loss


space_rf = [skopt.space.Integer(1, 200, name='n_estimators'),
         skopt.space.Integer(1, 80, name='max_depth'),
         skopt.space.Integer(1, 30, name='min_samples_leaf'),
         skopt.space.Integer(2, 20, name='min_samples_split'),
         skopt.space.Categorical(['auto', 'sqrt'], name='max_features')]

# space_keras = [skopt.space.]
# Add HP-space for Keras Regressor here


@skopt.utils.use_named_args(space_rf)
def objective_rf(**params):
    return train_evaluate_rf(X_train, y_train, X_val, y_val, params)

@skopt.utils.use_named_args(space_keras)
def objective_keras(**params):
    return train_evaluate_keras(X_train, y_train, X_val, y_val, params)


ALGORITHM = 'RandomForestRegressor' # 'RandomForestRegressor', 'Keras'
if ALGORITHM == 'RandomForestRegressor':
    thisObjective = objective_rf
    thisSpace = space_rf
elif ALGORITHM == 'Keras':
    thisObjective = objective_keras
    thisSpace = space_keras

OPTIMIZER = 'SMAC'  # 'GPBO', 'SMAC'
if OPTIMIZER == 'GPBO':
    res = gp_minimize(thisObjective, thisSpace, n_calls=100, random_state=0, acq_func='EI')
elif OPTIMIZER == 'SMAC':
    res = forest_minimize(thisObjective, thisSpace, n_calls=100, random_state=0, acq_func='EI')

print("Best score=%.4f" % res.fun)
