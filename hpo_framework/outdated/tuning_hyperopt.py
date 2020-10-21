from sklearn.ensemble import RandomForestRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from datasets.dummy import preprocessing as pp

FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)
# ML-algorithm
ALGORITHM = 'XGBRegressor'  # 'RandomForestRegressor', 'Keras', 'XGBRegressor'

# Define Hyperparameter-space for RandomForestRegressor
rf_space = {}
rf_space['n_estimators'] = hp.choice('n_estimators', range(1, 201, 1))
rf_space['max_depth'] = hp.choice('max_depth', range(1, 81, 1))
rf_space['min_samples_leaf'] = hp.choice('min_samples_leaf', range(1, 31, 1))
rf_space['min_samples_split'] = hp.choice('min_samples_split', range(2, 21, 1))
rf_space['max_features'] = hp.choice('max_features', ['auto', 'sqrt'])

# Define Hyperparameter-space for Keras-Regressor
keras_space = {}
keras_space['lr'] = hp.uniform('lr', low=1e-6, high=1e-1)
keras_space['dropout_rate'] = hp.uniform('dropout_rate', low=0.0, high=0.9)
keras_space['width_1stlayer'] = hp.choice('width_1stlayer', range(8, 513, 1))
keras_space['num_hidden_layers'] = hp.choice('num_hidden_layers', range(1, 5))
keras_space['width_hidlayer1'] = hp.choice('width_hidlayer1', range(10, 100, 10))
keras_space['width_hidlayer2'] = hp.choice('width_hidlayer2', range(10, 100, 10))
keras_space['width_hidlayer3'] = hp.choice('width_hidlayer3', range(10, 100, 10))
keras_space['width_hidlayer4'] = hp.choice('width_hidlayer4', range(10, 100, 10))
# >> Handle conditional hyperparameters https://github.com/hyperopt/hyperopt/wiki/FMin
# # only binary choices for conditional hyperparameter (hidden layer 1 (yes/no))
# keras_space['hidden_layer_no1'] = hp.choice('hidden_layer_no1', [
#     ('no', 0),
#     ('yes', hp.choice('width_hidlayer1', range(8, 513, 8)))
# ])
#
# keras_space['hidden_layer_no2'] = hp.choice('hidden_layer_no2', [
#     ('no', 0),
#     ('yes', hp.choice('width_hidlayer2', range(8, 257, 8)))
# ])

# Define Hyperparameter-space for XGBRegressor
xgb_space = {}
xgb_space['booster'] = hp.choice('booster', ['gbtree', 'gblinear', 'dart'])
xgb_space['n_estimators'] = hp.choice('n_estimators', range(1, 201, 1))
xgb_space['max_depth'] = hp.choice('max_depth', range(1, 81, 1))


def train_evaluate_rf(X_train, y_train, X_val, y_val, params):
    rf_reg = RandomForestRegressor(**params, random_state=0)

    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_val)

    val_loss = sqrt(mean_squared_error(y_val, y_pred))

    return val_loss


def train_evaluate_keras(X_train, y_train, X_val, y_val, params):  # Assign n_func_evals as the number of epochs (int)
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=len(X_train.keys())))

    model.add(keras.layers.Dense(params['width_1stlayer'], activation='relu'))

    # if params['hidden_layer_no1'][1] > 0:
    #     model.add(keras.layers.Dense(params['hidden_layer_no1'][1], activation='relu'))
    #
    # if params['hidden_layer_no1'][1] > 0 and params['hidden_layer_no2'][1] > 0:
    #     model.add(keras.layers.Dense(params['hidden_layer_no2'][1], activation='relu'))

    if params['num_hidden_layers'] > 0:
        model.add(keras.layers.Dense(params['width_hidlayer1'], activation='relu'))

    if params['num_hidden_layers'] > 1:
        model.add(keras.layers.Dense(params['width_hidlayer2'], activation='relu'))

    if params['num_hidden_layers'] > 2:
        model.add(keras.layers.Dense(params['width_hidlayer3'], activation='relu'))

    if params['num_hidden_layers'] > 3:
        model.add(keras.layers.Dense(params['width_hidlayer4'], activation='relu'))

    model.add(keras.layers.Dropout(params['dropout_rate']))
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.RMSprop(learning_rate=params['lr'])

    model.compile(optimizer=optimizer, loss='mse')

    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val), verbose=1)

    y_pred = model.predict(X_val)

    try:
        val_loss = sqrt(mean_squared_error(y_val, y_pred))
    except ValueError:
        print('Check the ranges of the hyperparameters')
        val_loss = 10e10  # Better error handling necessary!

    return val_loss

def train_evaluate_xgb(X_train, y_train, X_val, y_val, params):
    xgb_reg = XGBRegressor(**params, random_state=0)

    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_val)

    val_loss = sqrt(mean_squared_error(y_val, y_pred))

    return val_loss


# Objective functions to be minimized
def objective_rf(params):
    val_loss = train_evaluate_rf(X_train, y_train, X_val, y_val, params)
    return {'loss': val_loss,
            'status': STATUS_OK,
            'eval_time': time.time()}


def objective_keras(params):
    val_loss = train_evaluate_keras(X_train, y_train, X_val, y_val, params)
    return {'loss': val_loss,
            'status': STATUS_OK,
            'eval_time': time.time()}


def objective_xgb(params):
    val_loss = train_evaluate_xgb(X_train, y_train, X_val, y_val, params)
    return {'loss': val_loss,
            'status': STATUS_OK,
            'eval_time': time.time()}


trials = Trials()

if ALGORITHM == 'RandomForestRegressor':
    res = fmin(fn=objective_rf, space=rf_space, trials=trials, algo=tpe.suggest, max_evals=100)
elif ALGORITHM == 'Keras':
    res = fmin(fn=objective_keras, space=keras_space, trials=trials, algo=tpe.suggest, max_evals=100)
elif ALGORITHM == 'XGBRegressor':
    res = fmin(fn=objective_xgb, space=xgb_space, trials=trials, algo=tpe.suggest, max_evals=100)

print(res)

# Plot the learning curve
result_list = trials.results
best_loss_curve = []
time_list = []

for i in range(len(result_list)):

    if i == 0:
        best_loss_curve.append(result_list[i]['loss'])
    elif result_list[i]['loss'] < min(best_loss_curve):
        best_loss_curve.append(result_list[i]['loss'])
    else:
        best_loss_curve.append(min(best_loss_curve))

    time_list.append(result_list[i]['eval_time'])

fig, ax = plt.subplots()
ax.plot(time_list, best_loss_curve)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Loss')

plt.show()

# # Train best model and save submission file
# X = pd.concat(objs=[X_train, X_val], axis=0)
# y = pd.concat(objs=[y_train, y_val], axis=0)
#
# xgb_best = XGBRegressor(random_state=0, booster='dart', max_depth=22, n_estimators=98)
# xgb_best.fit(X, y)
#
# TEST_FILE = 'test.csv'
# submission_file = 'sample_submission.csv'
#
# sample_submission = pp.load_data(FOLDER, submission_file)
# submission = sample_submission.copy()
#
# y_pred = xgb_best.predict(X_test)
# submission['SalePrice'] = y_pred
# submission.to_csv('submission.csv')

