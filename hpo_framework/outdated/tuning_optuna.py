import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from optuna.samplers import TPESampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import RandomSampler
from tensorflow import keras
import matplotlib.pyplot as plt

from datasets.dummy import preprocessing as pp

# Loading data and preprocessing
FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)
# ML-algorithm
ALGORITHM = 'RandomForestRegressor'  # 'RandomForestRegressor', 'Keras'
# HPO-method
SAMPLER = 'TPE'  # 'TPE', 'CMA-ES'


# 1. Define an objective function to be maximized
# Random Forest Regressor
def rf_objective(trial):
    # 2. Suggest values for the hyperparameters using a trial object
    n_estimators = trial.suggest_int(name='n_estimators', low=1, high=200)
    max_depth = trial.suggest_int(name='max_depth', low=1, high=80)
    min_samples_leaf = trial.suggest_int(name='min_samples_leaf', low=1, high=30)
    min_samples_split = trial.suggest_int(name='min_samples_split', low=2, high=20)
    max_features = trial.suggest_categorical(name='max_features', choices=['auto', 'sqrt'])

    rf_reg = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                   max_features=max_features)

    rf_reg.fit(X_train, y_train)

    y_pred = rf_reg.predict(X_val)

    val_loss = sqrt(mean_squared_error(y_val, y_pred))

    return val_loss


# Keras Regressor
def keras_objective(trial):
    dropout_rate = trial.suggest_uniform(name='droput_rate', low=0.0, high=0.9)
    lr = trial.suggest_uniform(name='lr', low=1e-6, high=1e-1)
    width_1stlayer = trial.suggest_int(name='width_1stlayer', low=8, high=512, step=8)
    num_hidden_layers = trial.suggest_int(name='num_hidden_layers', low=1, high=4)

    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=len(X_train.keys())))

    model.add(keras.layers.Dense(width_1stlayer, activation='relu'))

    if num_hidden_layers > 0:
        width_hidlayer1 = trial.suggest_int(name='width_hidlayer1', low=10, high=100, step=10)
        model.add(keras.layers.Dense(width_hidlayer1, activation='relu'))

    if num_hidden_layers > 1:
        width_hidlayer2 = trial.suggest_int(name='width_hidlayer2', low=10, high=100, step=10)
        model.add(keras.layers.Dense(width_hidlayer2, activation='relu'))

    if num_hidden_layers > 2:
        width_hidlayer3 = trial.suggest_int(name='width_hidlayer3', low=10, high=100, step=10)
        model.add(keras.layers.Dense(width_hidlayer3, activation='relu'))

    if num_hidden_layers > 3:
        width_hidlayer4 = trial.suggest_int(name='width_hidlayer4', low=10, high=100, step=10)
        model.add(keras.layers.Dense(width_hidlayer4, activation='relu'))

    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='mse')

    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val), verbose=1)
    y_pred = model.predict(X_val)

    val_loss = sqrt(mean_squared_error(y_val, y_pred))

    return val_loss

# 3. Create a study object an optimize the objective function
if SAMPLER == 'TPE':
    thisSampler = TPESampler()
elif SAMPLER == 'CMA-ES':
    thisSampler = CmaEsSampler()
else:
    thisSampler = RandomSampler()

if ALGORITHM == 'RandomForestRegressor':
    thisObjective = rf_objective
elif ALGORITHM == 'Keras':
    thisObjective = keras_objective

study = optuna.create_study(sampler=thisSampler, direction='minimize')
study.optimize(func=thisObjective, n_trials=100)

# >> test how to access the trials data
all_trials = study.get_trials()  # List of Froze-Trial objects
df_trials = study.trials_dataframe()  # Pandas DataFrame

# Plot the learning curve
best_loss_curve = []
time_list = []
for i in range(len(all_trials)):
    if i == 0:
        best_loss_curve.append(all_trials[i].value)
    elif all_trials[i].value < min(best_loss_curve):
        best_loss_curve.append(all_trials[i].value)
    else:
        best_loss_curve.append(min(best_loss_curve))

    time_list.append(all_trials[i].datetime_complete)

fig, ax = plt.subplots()
plt.plot(time_list, best_loss_curve)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Loss')

plt.show()
