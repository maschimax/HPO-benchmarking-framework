import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.filters import sobel
import pandas as pd


def importNegativeImages(lTotal):
    hData = []
    for i in range(lTotal):
        i += 1
        hData.append(np.asarray(Image.open(
            "C:/Users/Max/Documents/GitHub/HPO-benchmarking-framework/datasets/Surface_Crack_Image/Negative/" + str(
                i).zfill(5) + ".jpg").convert(mode='L')))

    return hData


def importPositiveImages(lTotal):
    hData = []
    for i in range(lTotal):
        i += 1
        if i < 19379 and i > 9999:
            hData.append(np.asarray(
                Image.open(
                    "C:/Users/Max/Documents/GitHub/HPO-benchmarking-framework/datasets/Surface_Crack_Image/Positive/" + str(
                        i).zfill(5) + "_1.jpg").convert(
                    mode='L')))
        else:
            hData.append(np.asarray(
                Image.open(
                    "C:/Users/Max/Documents/GitHub/HPO-benchmarking-framework/datasets/Surface_Crack_Image/Positive/" + str(
                        i).zfill(5) + ".jpg").convert(
                    mode='L')))

    return hData


def surface_crack_loading_and_preprocessing():
    # Number of images per class (max. 20,000)
    images_per_class = 2000

    # Load raw images of both classes
    image_data = importNegativeImages(images_per_class) + importPositiveImages(images_per_class)

    # Apply Sobel filter to detect edges
    for i in range(len(image_data)):
        this_sobel_image = sobel(image_data[i])
        image_data[i] = this_sobel_image

    # Structure the data a numpy arrays
    X_data = np.asarray(image_data).reshape(2 * images_per_class, image_data[0].shape[0] * image_data[0].shape[1])

    # Create labels
    y_data = np.asarray(np.zeros(images_per_class).tolist() + np.ones(images_per_class).tolist())

    # Train/Test-split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, shuffle=True)

    # Convert numpy arrays to pandas DataFrames / Series
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    return X_train, X_test, y_train, y_test


########################################################################################################################
X_train, X_test, y_train, y_test = surface_crack_loading_and_preprocessing()

# Modeling

# from tensorflow import keras
#
# model = keras.Sequential()
#
# model.add(keras.layers.InputLayer(len(X_train.keys())))
# model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dense(1, activation='sigmoid'))
#
# adam = keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=adam, loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=0.2, shuffle=True, verbose=1)
#
# y_pred = model.predict(X_test)
#
# y_pred = np.rint(y_pred)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: ', accuracy)
