import skopt
from skopt.optimizer import gp_minimize
from skopt.optimizer import forest_minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow import keras

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
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=len(X_train.keys())))
    model.add(keras.layers.Dense(params['width_1stlayer'], activation='relu'))

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

    val_loss = sqrt(mean_squared_error(y_val, y_pred))
    return val_loss


space_rf = [skopt.space.Integer(1, 200, name='n_estimators'),
            skopt.space.Integer(1, 80, name='max_depth'),
            skopt.space.Integer(1, 30, name='min_samples_leaf'),
            skopt.space.Integer(2, 20, name='min_samples_split'),
            skopt.space.Categorical(['auto', 'sqrt'], name='max_features')]

space_keras = [skopt.space.Real(low=1e-6, high=1e-1, name='lr'),
               skopt.space.Real(low=0.0, high=0.9, name='dropout_rate'),
               skopt.space.Integer(low=8, high=512, name='width_1stlayer'),
               skopt.space.Integer(low=1, high=4, name='num_hidden_layers'),
               skopt.space.Integer(low=10, high=100, name='width_hidlayer1'),
               skopt.space.Integer(low=10, high=100, name='width_hidlayer2'),
               skopt.space.Integer(low=10, high=100, name='width_hidlayer3'),
               skopt.space.Integer(low=10, high=100, name='width_hidlayer4')]


@skopt.utils.use_named_args(space_rf)
def objective_rf(**params):
    return train_evaluate_rf(X_train, y_train, X_val, y_val, params)


@skopt.utils.use_named_args(space_keras)
def objective_keras(**params):
    return train_evaluate_keras(X_train, y_train, X_val, y_val, params)


ALGORITHM = 'Keras'  # 'RandomForestRegressor', 'Keras'
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
