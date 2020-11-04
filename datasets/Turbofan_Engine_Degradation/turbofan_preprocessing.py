import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(data_path):

    # Column names for DataFrames
    operational_settings = ['operational_setting_{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['sensor_measurement_{}'.format(i + 1) for i in range(26)]
    cols = ['engine_no', 'time_in_cycles'] + operational_settings + sensor_columns

    data = pd.read_csv(data_path, sep=' ', header=None, names=cols)

    return data


def identify_nan_cols(train_df):

    # Identify features with more than 80 % missing data (in the training set)
    nan_cols = [col for col in train_df.columns if (train_df[col].isna().sum() / len(train_df)) >= 0.8]

    return nan_cols


def identify_corr_cols(train_df, corr_threshold=0.95):

    # Compute the correlation matrix
    corr_matrix = train_df.corr().abs()

    # Select the upper triangle of the corelation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find the columns with a correlation greater than the threshold
    corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]

    return corr_cols


def turbofan_loading_and_preprocessing():

    # Load the datasets
    raw_train_df = load_data('./datasets/Turbofan_Engine_Degradation/train_FD001.txt')
    raw_test_df = load_data('./datasets/Turbofan_Engine_Degradation/test_FD001.txt')
    rul = pd.read_csv('./datasets/Turbofan_Engine_Degradation/RUL_FD001.txt', decimal=".", header=None, names=['RUL'])

    # Add RUL (Remaining Useful Lifetime) column to training DataFrame
    train_rul_df = raw_train_df.groupby(by='engine_no', as_index=False)['time_in_cycles'].max()
    train_df = pd.merge(raw_train_df, train_rul_df, how='left', on='engine_no')
    train_df['RUL'] = train_df['time_in_cycles_y'] - train_df['time_in_cycles_x']

    # Drop the added column (this feature is not included in the test_set)
    train_df.drop(labels=['time_in_cycles_y'], inplace=True, axis=1)

    # Use the original column name for the time in cycles
    train_df.rename(columns={'time_in_cycles_x': 'time_in_cycles'}, inplace=True)

    # Training features and labels
    x_train = train_df.copy(deep=True)
    x_train.drop(labels=['engine_no'], inplace=True, axis=1)
    y_train = x_train.pop('RUL')

    # Add RUL (Remaining Useful Lifetime) column to test DataFrame
    max_cycle_idx = []
    for this_engine in raw_test_df['engine_no'].unique():
        # engines.append(this_engine)
        max_cycle_idx.append(raw_test_df[raw_test_df['engine_no'] == this_engine]['time_in_cycles'].idxmax())

    # Test features and labels
    x_test = raw_test_df.iloc[max_cycle_idx]
    x_test.reset_index(drop=True, inplace=True)
    x_test.drop(labels=['engine_no'], inplace=True, axis=1)
    y_test = rul['RUL']

    # Drop NaN columns in training and test set
    nan_cols = identify_nan_cols(x_train)
    x_train.drop(labels=nan_cols, axis=1, inplace=True)
    x_test.drop(labels=nan_cols, axis=1, inplace=True)

    # Drop constant columns
    const_columns = [col for col in x_train.columns if x_train[col].std() < 0.0000001]
    x_train.drop(labels=const_columns, axis=1, inplace=True)
    x_test.drop(labels=const_columns, axis=1, inplace=True)

    # Drop correlated
    corr_columns = identify_corr_cols(x_train)
    x_train.drop(labels=corr_columns, axis=1, inplace=True)
    x_test.drop(labels=corr_columns, axis=1, inplace=True)

    # Scaling (don't scale 'time_in_cycles')
    scaler = MinMaxScaler()
    x_train.iloc[:, 1:] = scaler.fit_transform(x_train.iloc[:, 1:])
    x_test.iloc[:, 1:] = scaler.transform(x_test.iloc[:, 1:])

    return x_train, x_test, y_train, y_test

