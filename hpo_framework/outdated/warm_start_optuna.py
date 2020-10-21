import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

import os
from pathlib import Path
from datasets.dummy import preprocessing as pp

###
# Preprocessing
abs_folder_path = os.path.abspath(path='datasets')
data_folder = Path(abs_folder_path)
train_file = "train.csv"
test_file = "test.csv"
submission_file = "sample_submission.csv"

train_raw = pp.load_data(data_folder, train_file)
test_raw = pp.load_data(data_folder, test_file)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)
###


def wst_objective(trial):
    n_est = trial.suggest_int(name='n_estimators', low=100, high=100)
    m_depth = trial.suggest_int(name='max_depth', low=40, high=40)

    params = {'n_estimators': n_est,
              'max_depth': m_depth}

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    loss = sqrt(mean_squared_error(y_val, y_pred))

    return loss


def objective(trial):
    n_est = trial.suggest_int(name='n_estimators', low=1, high=200)
    m_depth = trial.suggest_int(name='max_depth', low=1, high=80)

    params = {'n_estimators': n_est,
              'max_depth': m_depth}

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    loss = sqrt(mean_squared_error(y_val, y_pred))

    return loss


optimizer = TPESampler()

wst_study_name = 'wst-study'

print('Warmstart')

wst_study = optuna.create_study(sampler=optimizer, study_name=wst_study_name, storage='sqlite:///wst1.db')
wst_study.optimize(func=wst_objective, n_trials=1)

print('Optimization')

opt_study = optuna.create_study(sampler=optimizer, study_name=wst_study_name, storage='sqlite:///wst1.db',
                                load_if_exists=True)
opt_study.optimize(func=objective, n_trials=10)


