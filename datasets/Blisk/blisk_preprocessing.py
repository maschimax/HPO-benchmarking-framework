import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time


def blisk_loading_and_preprocessing(sample_data=True, sampling_rate=34):

    file_path = './datasets/Blisk/SL_S7_B2_KB1_prepared.csv'

    # Load the data set
    start_loading = time.time()
    X_data = pd.read_csv(file_path, sep='\t', index_col=0)
    loading_time = time.time() - start_loading
    print('Loading duration [s]:', loading_time)

    # Sample the original time series with the sampling rate to reduce the data set size
    if sample_data:

        sample_idx = list(range(0, len(X_data), sampling_rate))
        X_data = X_data.iloc[sample_idx, :]

    # fig, ax = plt.subplots()
    # ax.plot(X_data.index, X_data['vibration[t+1]'])
    # plt.show()

    X_data.reset_index(drop=True, inplace=True)

    # Separate labels
    y_data = X_data.pop('vibration[t+1]')

    # Train-test-split WITHOUT shuffling
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)

    # Scaling with scikit-learn's StandardScaler
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    return X_train, X_test, y_train, y_test
