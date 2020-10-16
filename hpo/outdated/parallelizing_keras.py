import json
import os
import sys
import preprocessing as pp
from pathlib import Path
from hpo.outdated.keras_sequence_testing import MySequence

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ.pop('TF_CONFIG', None)

if '.' not in sys.path:
    sys.path.insert(0, '.')

import tensorflow as tf
from tensorflow import keras

# Preprocessing
abs_folder_path = os.path.abspath(path='/home/max/Desktop/Projects/housing_regression/datasets')
data_folder = Path(abs_folder_path)
train_file = "train.csv"
test_file = "test.csv"
submission_file = "sample_submission.csv"

train_raw = pp.load_data(data_folder, train_file)
test_raw = pp.load_data(data_folder, test_file)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)

train_seq = MySequence(X_train, y_train, batch_size=64)
test_seq = MySequence(X_val, y_val, batch_size=64)


def build_and_compile_keras_regressor(x_train):
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=len(x_train.keys())))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dense(units=32, activation='relu'))
    model.add(keras.layers.Dense(units=16, activation='relu'))

    model.add(keras.layers.Dense(units=1, activation='linear'))

    adam = keras.optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')

    return model


this_model = build_and_compile_keras_regressor(X_train)
this_model.fit(x=train_seq, epochs=1000, batch_size=64, validation_data=(X_val, y_val), workers=4,
               use_multiprocessing=True)
bla = 0
