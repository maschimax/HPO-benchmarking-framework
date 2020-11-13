from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from math import sqrt
import numpy as np


def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


def f1_loss(y_true, y_pred):
    loss = 1 - f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    return loss


def accuracy_loss(y_true, y_pred):
    loss = 1 - accuracy_score(y_true, y_pred)
    return loss


def rul_loss_score(y_true, y_pred):
    # Loss metric based on the deviation from the expected to the true RUL (Remaining Useful Lifetime)
    # Special loss metric for Turbofan Engine Degradation use case
    # Reference: https://ieeexplore.ieee.org/document/4711414
    h = (y_pred - y_true)
    score = [(np.exp(-x/13)-1) if x < 0 else (np.exp(x/10)-1) for x in h]
    loss = np.sum(score)

    return loss


def area_under_curve(mean_trace_desc: list, lower_bound=0.0):
    """
    Function computes the area under curve (learning curve), which is a measure for the speed / cost of HPO
    :param mean_trace_desc: list
        List of descending, mean loss values of a trial.
    :param lower_bound: float
        Lower bound to ensure that the AUC ist always positive (0.0 for common loss measures)
    :return:
        Area under curve (AUC)
    """
    num_evals = len(mean_trace_desc)

    auc = 1/num_evals * sum(loss_val - lower_bound for loss_val in mean_trace_desc)

    return auc

