import pandas as pd


class Optimizer:
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

    def optimize(self):

        raise NotImplementedError

    def get_best_configuration(self):
        # returns the best configuration as a dictionary
        raise NotImplementedError

    def get_best_score(self):
        # returns the validation score of the best configuration
        raise NotImplementedError

    def plot_learning_curve(self):

        raise NotImplementedError

    def get_metrics(self):

        raise NotImplementedError
