from sklearn.metrics import mean_squared_error
from math import sqrt


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
