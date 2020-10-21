import lightgbm as lgb
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error
from math import sqrt

from datasets.dummy import preprocessing as pp

# Preprocessing
abs_folder_path = os.path.abspath(path='/home/max/Desktop/Projects/housing_regression/datasets/dummy')
data_folder = Path(abs_folder_path)
train_file = "train.csv"
test_file = "test.csv"
submission_file = "sample_submission.csv"

train_raw = pp.load_data(data_folder, train_file)
test_raw = pp.load_data(data_folder, test_file)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)

train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_val, label=y_val)

param = {'num_leaves': 31,
         'taks': 'regression',
         'metric': 'rmse',
         'seed': 0}

bst = lgb.train(params=param, train_set=train_data, num_boost_round=200, valid_sets=[validation_data])

y_pred = bst.predict(data=X_val)

val_loss = sqrt(mean_squared_error(y_val, y_pred))

print('Validation loss: ', val_loss)
