import pickle
import pandas as pd
from imblearn.over_sampling import ADASYN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def sensor_loading():
    folder = './datasets/Sensor_System_Production'

    # Load preprocessed data set
    X = pickle.load(open(folder + "/pickle_data/X.p", "rb"))
    X_new = pickle.load(open(folder + "/pickle_data/X_new.p", "rb"))
    y_process_labels = pickle.load(open(folder + "/pickle_data/y_process_labels.p", "rb"))

    # Only use a single subset
    X_new = X_new[4]
    y_process_labels = pd.Series(y_process_labels[4])

    return X_new, y_process_labels


def sensor_balancing(X_train, y_train):

    # Drop all rows with a very rare results, since SMOTEEN cannot handle them
    cc = y_train.value_counts()[y_train.value_counts() <= 3]
    y_train = y_train[~y_train.isin(cc.index.values)]
    X_train = pd.DataFrame(X_train[X_train.index.isin(list(y_train.index))])

    y_train = pd.Series(y_train)
    columns = pd.DataFrame(X_train).columns.values

    # Perform oversampling
    adasyn = ADASYN(sampling_strategy='not majority', n_neighbors=2, n_jobs=1)
    # X_train, y_train = adasyn.fit_sample(X_train, np.ravel(y_train.values))
    X_train, y_train = adasyn.fit_sample(X_train, y_train)

    X_train = pd.DataFrame(X_train, columns=list(columns))
    return X_train, pd.Series(y_train)


def sensor_loading_and_preprocessing():
    # Load data set
    X_raw, y_raw = sensor_loading()

    # Label encoding of y-vector
    lab_enc = LabelEncoder()
    y_raw = pd.Series(lab_enc.fit_transform(y_raw))

    # Train-test-split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=0, shuffle=True)

    # Balance the training data
    X_train, y_train = sensor_balancing(X_train, y_train)

    return X_train, X_test, y_train, y_test
