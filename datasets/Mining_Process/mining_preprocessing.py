import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def identify_const_columns(train_df):
    const_cols = [col for col in train_df.columns if train_df[col].std() < 0.0000001]
    return const_cols


def identify_corr_cols(train_df, corr_threshold=0.95):
    # Compute the correlation matrix
    corr_matrix = train_df.corr().abs()

    # Select the upper triangle of the corelation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find the columns with a correlation greater than the threshold
    corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]

    return corr_cols


def mining_loading_and_preprocessing(test_split=0.2):
    # Loading the data
    raw_df = pd.read_csv('./datasets/Mining_Process/MiningProcess_Flotation_Plant_Database.csv', decimal=',')

    # Separate the Labels
    y_raw = raw_df.pop('% Silica Concentrate')

    # Drop iron concentrate column, since it is the task to predict the Silica concentrate without knowing
    # the iron concentrate
    raw_df.drop(labels='% Iron Concentrate', axis=1, inplace=True)

    # Drop date column
    raw_df.drop(labels='date', axis=1, inplace=True)

    # Shuffle and split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(raw_df, y_raw, test_size=test_split, random_state=0,
                                                        shuffle=True)

    # Drop constant cols
    const_cols = identify_const_columns(x_train)
    x_train.drop(labels=const_cols, axis=1, inplace=True)
    x_test.drop(labels=const_cols, axis=1, inplace=True)

    # Drop correlated cols
    corr_cols = identify_corr_cols(x_train)
    x_train.drop(labels=corr_cols, axis=1, inplace=True)
    x_test.drop(labels=corr_cols, axis=1, inplace=True)

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = pd.DataFrame(scaler.fit_transform(x_train))
    x_test = pd.DataFrame(scaler.transform(x_test))

    # PCA?

    return x_train, x_test, y_train, y_test
