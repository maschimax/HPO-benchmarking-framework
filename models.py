#%%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from preprocessing import process
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

FOLDER = 'datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

def load_data(folder, file):
    filename = os.path.join(folder, file)
    return pd.read_csv(filename, index_col = 'Id')

train_raw = load_data(FOLDER, TRAIN_FILE)
test_raw = load_data(FOLDER, TEST_FILE)
sample_submission = load_data(FOLDER, SAMPLE_SUB)
submission = sample_submission.copy()

print("train_raw.shape: ", train_raw.shape)
print("test_raw.shape: ", test_raw.shape)
#%%
fig, ax = plt.subplots()

distributions = [stats.norm, stats.lognorm, stats.johnsonsu]
for dist in distributions:
    sns.distplot(train_raw['SalePrice'], kde=False, fit=dist, axlabel='SalePrice - '+str(dist))
    plt.show()

#%%
# total_cols = [col for col in train_raw.columns]
# cat_cols = [col for col in train_raw.columns if train_raw[col].dtype == 'object']
# num_cols = list(set(total_cols)-set(cat_cols))

# for col in num_cols:
#     sns.distplot(train_raw[col], kde=False)
#     plt.show()
#%%
# PREPROCESSING
X_train, y_train, X_val, y_val, X_test = process(train_raw, test_raw, standardization=False, logarithmic=False, count_encoding=False)

#%%
# Random Forest Regressor
ESTIMATORS = [220, 222, 224, 226, 228, 230]
rf_scores = {}

for est in ESTIMATORS:
    rf_reg = RandomForestRegressor(n_estimators=est, random_state=0, n_jobs=4)
    rf_reg.fit(X_train, y_train)
    predictions=rf_reg.predict(X_val)
    rf_scores[est] = sqrt(mean_squared_error(y_val, predictions))

print("Random Forest - best number of estimators: ", min(rf_scores, key=rf_scores.get))
print("Random Forest Regressor validation RMSE (Manual Preprocessing): ", rf_scores[min(rf_scores, key=rf_scores.get)])

# %%
# XGBoost Regressor 
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=4)

xgb_reg.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=0) 

xgb_predictions = xgb_reg.predict(X_val)

xgb_score = sqrt(mean_squared_error(y_val, xgb_predictions))
print("XGBoost validation RMSE (Manual Preprocessing): ", xgb_score)

#%%
# xgb_predictions_test = xgb_reg.predict(X_test)
# submission['SalePrice'] = xgb_predictions_test
# submission.to_csv('xgb_submission.csv')

# %%
# Neural net with Keras
nn_model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=[len(X_train.keys())], kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.RMSprop(0.001)

nn_model.compile(optimizer=optimizer, loss='mse')

EPOCHS = 250
nn_model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0)

nn_predictions = nn_model.predict(X_val)
nn_score = sqrt(mean_squared_error(y_val, nn_predictions))
print("Neural Network validation RMSE (Manual Preprocessing): ", nn_score)

#%%
# nn_predictions_test = nn_model.predict(X_test)
# submission['SalePrice'] = nn_predictions_test
# submission.to_csv('nn_submission.csv')
#%%
'''
# PREPROCESSING 2 (Pipeline)
# Pipeline for NaN-Handling and OneHotEncoding
numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('OHencoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

#%%
ESTIMATORS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
rf_scores = {}

for est in ESTIMATORS:
    rf_reg = RandomForestRegressor(n_estimators=est, random_state=0)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_reg)
    ])
    pipeline.fit(X_train, y_train)
    predictions=pipeline.predict(X_val)
    rf_scores[est] = sqrt(mean_squared_error(y_val, predictions))

print("best number of estimators: ", min(rf_scores, key=rf_scores.get))
print("RMSE: ", rf_scores[min(rf_scores, key=rf_scores.get)])
#%%
# BEST Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=225, random_state=0)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_reg)
])

pipeline.fit(X_train, y_train)

predictions=pipeline.predict(X_val)
rf_score = sqrt(mean_squared_error(y_val, predictions))
print("Random Forest validation RMSE: ", rf_score)
#%%
# XGBoost Regressor with pipeline
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.01)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_reg)
])
pipeline.fit(X_train, y_train)

predictions=pipeline.predict(X_val)
xgb_score = sqrt(mean_squared_error(y_val, predictions))
print("XGBoost validation RMSE (Pipeline): ", xgb_score)
'''