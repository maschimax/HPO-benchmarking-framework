import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE


def identify_const_columns(train_df):
    const_cols = [col for col in train_df.columns if train_df[col].std() < 0.0000001]
    return const_cols


def identify_corr_cols(train_df, corr_threshold=0.95):
    # Compute the correlation matrix
    corr_matrix = train_df.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find the columns with a correlation greater than the threshold
    corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]

    return corr_cols


def oversample_minority(x_imb, y_imb):

    # Initialize a SMOTE instance which performs synthetic minority over-sampling
    oversampler = SMOTE(sampling_strategy='not majority', random_state=0)

    # Do resampling
    x_bal, y_bal = oversampler.fit_resample(np.array(x_imb), np.array(y_imb))

    # Shuffle again
    x_bal, y_bal = shuffle(x_bal, y_bal, random_state=0)

    return pd.DataFrame(x_bal), pd.DataFrame(y_bal)


def steel_loading_and_preprocessing(test_split=0.2):
    # Loading the data
    raw_df = pd.read_csv('./datasets/Faulty_Steel_Plates/faults.csv')

    # Separate the labels
    label_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    y_raw = raw_df[label_cols].copy(deep=True)
    raw_df.drop(labels=label_cols, axis=1, inplace=True)

    # y_data = y_raw.loc[:, 'Fault_ID']
    y_data = y_raw.copy(deep=True)

    # Drop constant columns
    const_cols = identify_const_columns(raw_df)
    raw_df.drop(const_cols, axis=1, inplace=True)

    # Drop correlated columns
    corr_cols = identify_corr_cols(raw_df)
    raw_df.drop(corr_cols, axis=1, inplace=True)

    # Shuffle and split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(raw_df, y_data, test_size=test_split, random_state=0,
                                                        shuffle=True)

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = pd.DataFrame(scaler.fit_transform(x_train))
    x_test = pd.DataFrame(scaler.transform(x_test))

    # PCA?

    # Oversampling
    over = True
    if over:
        x_train, y_train = oversample_minority(x_train, y_train)

    return x_train, x_test, y_train, y_test
