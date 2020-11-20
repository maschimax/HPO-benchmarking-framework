import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def drop_correlated(df, corr_threshold=0.95):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find the columns with a correlation greater than the threshold
    drop_cols = [column for column in upper.columns if any(upper[column] > corr_threshold)]

    # return columns to drop
    return drop_cols


def undersample(df_X, df_y):
    # Use this function for undersampling the majority class

    # Get the number of positive labels
    num_pos = len(df_y[df_y == 1])

    # Get a list of indices of rows with neg labels
    indices_neg = df_y[df_y == 0].index

    # Choose randomly a number of values from the neg list (Undersample the negative class)
    random_indices = np.random.choice(indices_neg, num_pos, replace=False)

    # Get the list of indices with pos values to use
    indices_pos = df_y[df_y == 1].index

    # List with the undersample indices
    under_sample_indices = np.concatenate([indices_pos, random_indices])

    # Index based extraction of the undersampled dataset
    X_undersample = df_X.loc[under_sample_indices]
    y_undersample = df_y.loc[under_sample_indices]

    return X_undersample, y_undersample


def sample_SMOTEENN(df_X, df_y):

    # Initialize an SOMTEEN object to perform oversampling using SMOTE and cleaning using ENN
    sme = SMOTEENN(
        sampling_strategy=1,
        smote=SMOTE(sampling_strategy=0.3, k_neighbors=3, n_jobs=1, random_state=1),
        enn=EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=5, n_jobs=1),
        random_state=1)

    # Do resampling
    X_res, y_res = sme.fit_resample(df_X, df_y)

    # unique, counts = np.unique(y_res, return_counts=True)
    # print(X_res.shape, y_res.shape, dict(zip(unique, counts)))

    return X_res, y_res


def preprocess_scania(train_data, test_data, test_size):
    # https://www.kaggle.com/gxkok21/aps-failure-in-scania-trucks-challenge
    # https://www.kaggle.com/romulomadu/minimizing-total-cost-result-9020-00
    # https://www.kaggle.com/kagglespabr/scania-data-analysis

    # Concatenate both data sets
    raw_data = pd.concat(objs=[train_data, test_data], axis=0)

    # Encode class labels as integers
    raw_data['class'] = (raw_data['class'] == 'pos').astype('int')

    # Pop labels
    y_raw = raw_data.pop('class')

    X_train, X_test, y_train, y_test = train_test_split(raw_data, y_raw, test_size=test_size, random_state=0,
                                                        shuffle=True)

    # Impute NaN's with mean values
    nan_imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(nan_imputer.fit_transform(X_train))
    X_test = pd.DataFrame(nan_imputer.transform(X_test))

    # Drop constant columns
    const_columns = [col for col in X_train.columns if X_train[col].std() < 0.0000001]
    X_train.drop(const_columns, axis=1, inplace=True)
    X_test.drop(const_columns, axis=1, inplace=True)

    # Drop highly correlated columns / features
    drop_cols = drop_correlated(X_train)
    X_train.drop(drop_cols, axis=1, inplace=True)
    X_test.drop(drop_cols, axis=1, inplace=True)

    # Scaling the data between -1 and 1
    # Necessary for PCA, SVM, not with Trees, but the results shouldn't change
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    # Principal component analysis
    pca = PCA(0.95)
    X_train = pd.DataFrame(pca.fit_transform(X_train))
    X_test = pd.DataFrame(pca.transform(X_test))

    return X_train, X_test, y_train, y_test


def balance_scania(X_train_std_pca, y_train):

    print('Before using over- and undersampling, we have {0} negatives and {1} positives in our Training Data'.format(
        np.count_nonzero(y_train.values == 0), np.count_nonzero(y_train.values == 1)))

    # Oversampling of minority class
    X_train_smoteenn, y_train_smoteenn = sample_SMOTEENN(X_train_std_pca, y_train)

    df_X_smoteenn = pd.DataFrame(X_train_smoteenn)
    df_y_smoteenn = pd.Series(y_train_smoteenn)

    # Undersampling of majority class
    X_train_balanced, y_train_balanced = undersample(df_X_smoteenn, df_y_smoteenn)

    print('After using over- and undersampling, we have {0} negatives and {1} positives in our Training Data'.format(
        np.count_nonzero(y_train_balanced.values == 0), np.count_nonzero(y_train_balanced.values == 1)))

    # Shuffle again
    X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced, random_state=0)

    return X_train_balanced, y_train_balanced


def scania_loading_and_preprocessing(test_size=0.2):
    # Load datasets
    train_data = pd.read_csv("./datasets/Scania_APS_Failure/aps_failure_training_set.csv", na_values=["na"])
    test_data = pd.read_csv("./datasets/Scania_APS_Failure/aps_failure_test_set.csv", na_values=["na"])

    # Preprocessing (NaN-Handling, drop constant features, drop highly correlated features, standardization, PCA)
    X_train, X_test, y_train, y_test = preprocess_scania(train_data, test_data, test_size)

    # Balance the training dataset using a combination of over- and undersampling
    X_train_bal, y_train_bal = balance_scania(X_train, y_train)

    return X_train_bal, X_test, y_train_bal, y_test
