import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.filters import sobel
from skimage.transform import resize
import pandas as pd


def importNegativeImages(lTotal):
    hData = []
    for i in range(lTotal):
        i += 1
        hData.append(np.asarray(Image.open(
            "./datasets/Surface_Crack_Image/Negative/" + str(
                i).zfill(5) + ".jpg").convert(mode='L')))

    return hData


def importPositiveImages(lTotal):
    hData = []
    for i in range(lTotal):
        i += 1
        if 19379 > i > 9999:
            hData.append(np.asarray(
                Image.open(
                    "./datasets/Surface_Crack_Image/Positive/" + str(
                        i).zfill(5) + "_1.jpg").convert(
                    mode='L')))
        else:
            hData.append(np.asarray(
                Image.open(
                    "./datasets/Surface_Crack_Image/Positive/" + str(
                        i).zfill(5) + ".jpg").convert(
                    mode='L')))

    return hData


def surface_crack_loading_and_preprocessing(images_per_class=2000):
    # Max. number of images per class -> 20,000

    # Load raw images of both classes
    image_data = importNegativeImages(images_per_class) + importPositiveImages(images_per_class)

    # Apply Sobel filter to detect edges
    for i in range(len(image_data)):
        this_sobel_image = sobel(image_data[i])
        image_data[i] = this_sobel_image

    # Resize images to (64, 64) to decrease the number of features and accelerate the training process
    for i in range(len(image_data)):
        this_down_scaled_image = resize(image_data[i], output_shape=(64, 64), anti_aliasing=True)
        image_data[i] = this_down_scaled_image

    # Structure the data as numpy arrays
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

    print('Surface data set - number of training samples: ', str(len(y_train)))
    print('Surface data set - number of test samples: ', str(len(y_test)))

    return X_train, X_test, y_train, y_test
