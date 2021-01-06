import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


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


def identify_corr_cols(train_df, corr_threshold=0.95, show_heat_map=False):

    # Compute the correlation matrix
    corr_matrix = train_df.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find the columns with a correlation greater than the threshold
    corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]

    if show_heat_map:
        # Initialize figure the correlation matrix
        fig, ax = plt.subplots(figsize=(11, 9))

        # Set a seaborn theme
        sns.set_theme(style="white")

        # Create an ipt color map
        cmap = sns.light_palette(color='#179c7d', n_colors=10, as_cmap=True)

        # Mask to select the lower triangle of the correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot the heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=0.0, center=0.8,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.savefig('turbofan_corr_matrix.svg', bbox_inches='tight')
        plt.savefig('turbofan_corr_matrix.jpg', bbox_inches='tight')

    return corr_cols


def turbofan_loading_and_preprocessing():

    show_hist = False
    show_heat_map = False

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

    # Join x- and y- data sets in preparation of the split
    x_data = pd.concat(objs=[x_train, x_test], axis=0, ignore_index=True)
    y_data = pd.concat(objs=[y_train, y_test], axis=0, ignore_index=True)

    # 80 / 20 split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0, shuffle=True)

    if show_hist:
        # Create histogram for the label
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.set_theme(style="white")
        sns.histplot(y_train, stat='count', shrink=0.8, kde=True, color='#179c7d')
        plt.savefig('turbofan_label_histplot.jpg', bbox_inches='tight')
        plt.savefig('turbofan_label_histplot.svg', bbox_inches='tight')

    # # Visualize features with NaN-values
    # fig, ax = plt.subplots(figsize=(11, 9))
    # nan_cols = []
    # nan_shares = []
    # for col in x_train.columns:
    #     num_nan = x_train[col].isna().sum()
    #     if num_nan > 0:
    #         nan_percentage = num_nan / len(x_train[col]) * 100
    #         nan_cols.append(col)
    #         nan_shares.append(nan_percentage)
    #
    # sns.set_theme(style='white')
    # sns.barplot(x=nan_cols, y=nan_shares, color='#179c7d', saturation=0.8)
    # ax.set_ylabel('Share of missing values [%]')
    # plt.savefig('turbofan_nan_barplot.jpg')
    # plt.savefig('turbofan_nan_barplot.svg')

    # Drop NaN columns in training and test set
    nan_cols = identify_nan_cols(x_train)
    x_train.drop(labels=nan_cols, axis=1, inplace=True)
    x_test.drop(labels=nan_cols, axis=1, inplace=True)

    # Drop constant columns
    const_columns = [col for col in x_train.columns if x_train[col].std() < 0.0000001]
    x_train.drop(labels=const_columns, axis=1, inplace=True)
    x_test.drop(labels=const_columns, axis=1, inplace=True)

    # Drop correlated
    corr_columns = identify_corr_cols(x_train, show_heat_map=show_heat_map)
    x_train.drop(labels=corr_columns, axis=1, inplace=True)
    x_test.drop(labels=corr_columns, axis=1, inplace=True)

    # Scaling (don't scale 'time_in_cycles')
    scaler = MinMaxScaler()
    x_train.iloc[:, 1:] = scaler.fit_transform(x_train.iloc[:, 1:])
    x_test.iloc[:, 1:] = scaler.transform(x_test.iloc[:, 1:])

    return x_train, x_test, y_train, y_test

