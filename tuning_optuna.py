import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from optuna.samplers import TPESampler
from optuna.samplers import CmaEsSampler

import preprocessing as pp

# Loading data and preprocessing
FOLDER = 'datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)


# 1. Define an objective function to be maximized
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


# 3. Create a study object an optimize the objective function
SAMPLER = 'CMA-ES' # 'TPE', 'CMA-ES'

if SAMPLER == 'TPE':
    thisSampler = TPESampler()
elif SAMPLER == 'CMA-ES':
    thisSampler = CmaEsSampler()

ALGORITHM = 'RandomForestRegressor' # 'RandomForestRegressor', 'Keras'

if ALGORITHM == 'RandomForestRegressor':
    thisObjective = rf_objective

study = optuna.create_study(sampler=thisSampler, direction='minimize',)
study.optimize(func=rf_objective, n_trials=100)
