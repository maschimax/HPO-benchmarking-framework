import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid

from hpo.optuna_optimizer import OptunaOptimizer
from hpo.skopt_optimizer import SkoptOptimizer


class Trial:
    def __init__(self, hp_space: list, ml_algorithm: str, optimization_schedule: list, metric,
                 n_runs: int, budget: int, n_workers: int,
                 x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series):
        self.hp_space = hp_space
        self.ml_algorithm = ml_algorithm
        self.optimization_schedule = optimization_schedule
        self.metric = metric
        self.n_runs = n_runs
        self.budget = budget
        self.n_workers = n_workers
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        # Attribute for CPU / GPU selection required

    def run(self):

        # Process the optimization schedule
        results_dict = {}
        for opt_tuple in self.optimization_schedule:
            this_hpo_library = opt_tuple[0]
            this_hpo_method = opt_tuple[1]

            results_df = pd.DataFrame(columns=['HPO-library', 'HPO-method', 'ML-algorithm', 'run_id', 'random_seed',
                                               'num_of_evaluation', 'losses', 'timestamps'])

            # Perform n_runs with varying random seeds
            for i in range(self.n_runs):
                run_id = str(uuid.uuid4())
                this_seed = i  # Random seed for this run

                if this_hpo_library == 'skopt':
                    optimizer = SkoptOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                               ml_algorithm=self.ml_algorithm,
                                               x_train=self.x_train, x_val=self.x_val, y_train=self.y_train,
                                               y_val=self.y_val,
                                               metric=self.metric, budget=self.budget, random_seed=this_seed)
                elif this_hpo_library == 'optuna':
                    optimizer = OptunaOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                ml_algorithm=self.ml_algorithm,
                                                x_train=self.x_train, x_val=self.x_val, y_train=self.y_train,
                                                y_val=self.y_val,
                                                metric=self.metric, budget=self.budget, random_seed=this_seed)
                else:
                    raise NameError('Unknown HPO-library!')

                optimization_results = optimizer.optimize()

                temp_dict = {'HPO-library': [this_hpo_library] * len(optimization_results.losses),
                                'HPO-method': [this_hpo_method] * len(optimization_results.losses),
                                'ML-algorithm': [self.ml_algorithm] * len(optimization_results.losses),
                                'run_id': [run_id] * len(optimization_results.losses),
                                'random_seed': [i] * len(optimization_results.losses),
                                'num_of_evaluation': list(range(1, len(optimization_results.losses) + 1)),
                                'losses': optimization_results.losses,
                                'timestamps': optimization_results.timestamps}

                this_df = pd.DataFrame.from_dict(data=temp_dict)
                results_df = pd.concat(objs=[results_df, this_df], axis=0)

            results_dict[opt_tuple] = results_df

        return results_dict

    def plot_learning_curve(self, results_dict: dict):
        # Rework required
        fig, ax = plt.subplots()

        n_rows = len(results_dict[list(results_dict.keys())[0]].trial_ids)  # number of evaluations
        n_cols = len(results_dict)
        best_losses = np.zeros(shape=(n_rows, n_cols))

        for j in range(n_cols):
            for i in range(n_rows):

                if i == 0:
                    best_losses[i, j] = results_dict[list(results_dict.keys())[j]].losses[i]
                elif results_dict[list(results_dict.keys())[j]].losses[i] < best_losses[i - 1, j]:
                    best_losses[i, j] = results_dict[list(results_dict.keys())[j]].losses[i]
                else:
                    best_losses[i, j] = best_losses[i - 1, j]

        bla = 0
        mean_curve = np.mean(best_losses, axis=1)
        quant25_curve = np.quantile(best_losses, q=.25, axis=1)
        quant75_curve = np.quantile(best_losses, q=.75, axis=1)

        plt.plot(results_dict[list(results_dict.keys())[0]].timestamps, mean_curve)
        plt.plot(results_dict[list(results_dict.keys())[0]].timestamps, quant25_curve)
        plt.plot(results_dict[list(results_dict.keys())[0]].timestamps, quant75_curve)

        pass

    def get_metrics(self):
        pass
