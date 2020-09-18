import pandas as pd
from abc import ABC, abstractmethod

from hpo.results import TuningResult


class BaseOptimizer(ABC):
    def __init__(self, hp_space, hpo_method: str, ml_algorithm: str,
                 x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
                 metric, budget: int):
        """

        :param hp_space:
        :param hpo_method:
        :param ml_algorithm:
        :param x_train:
        :param x_val:
        :param y_train:
        :param y_val:
        :param metric
        :param budget:
        """

        self.hp_space = hp_space
        self.hpo_method = hpo_method
        self.ml_algorithm = ml_algorithm
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.metric = metric
        self.budget = budget

    @abstractmethod
    def optimize(self) -> TuningResult:

        raise NotImplementedError

    @staticmethod
    def get_best_configuration(result: TuningResult):
        # Returns the best configuration as a dictionary
        return result.best_configuration

    @staticmethod
    def get_best_score(result: TuningResult):
        # Returns the validation score of the best configuration
        raise result.best_loss

    @staticmethod
    def plot_learning_curve(result: TuningResult):

        raise NotImplementedError

    @staticmethod
    def get_metrics(result: TuningResult):

        raise NotImplementedError
