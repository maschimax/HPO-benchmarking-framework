import pandas as pd

from hpo.optuna_optimizer import OptunaOptimizer
from hpo.skopt_optimizer import SkoptOptimizer


class Trial:
    def __init__(self, hp_space: list, ml_algorithm: str, hpo_library: str, hpo_method: str, metric,
                 n_runs: int, budget: int, n_workers: int, seed_variation: bool,
                 x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series):
        self.hp_space = hp_space
        self.ml_algorithm = ml_algorithm
        self.hpo_library = hpo_library
        self.hpo_method = hpo_method
        self.metric = metric
        self.n_runs = n_runs
        self.budget = budget
        self.n_workers = n_workers
        self.seed_variation = seed_variation
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        # Attribute for CPU / GPU selection required

    def run(self):
        if self.hpo_library == 'skopt':
            optimizer = SkoptOptimizer(hp_space=self.hp_space, hpo_method=self.hpo_method,
                                       ml_algorithm=self.ml_algorithm,
                                       x_train=self.x_train, x_val=self.x_val, y_train=self.y_train, y_val=self.y_val,
                                       metric=self.metric, budget=self.budget)
        elif self.hpo_library == 'optuna':
            optimizer = OptunaOptimizer(hp_space=self.hp_space, hpo_method=self.hpo_method,
                                        ml_algorithm=self.ml_algorithm,
                                        x_train=self.x_train, x_val=self.x_val, y_train=self.y_train, y_val=self.y_val,
                                        metric=self.metric, budget=self.budget)
        else:
            raise NameError('Unknown HPO-library!')

        # >>> Implement the number of runs with a random seed variation

        optimization_results = optimizer.optimize()
        return optimization_results
