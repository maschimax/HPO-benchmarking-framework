import numpy as np
from robo.fmin import bayesian_optimization, fabolas

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class RoboOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                         random_seed)

        def optimize(self) -> TuningResult:
            pass

        def objective(self, cont_hp_space, **kwargs):
            """

            :param self:
            :param cont_hp_space: np.array
            array that contains the next hyperparameter configuration (continuous) to be evaluated
            :param kwargs:
            :return:
            """

            if 's' in kwargs:   # Fabolas
                s = kwargs['s']
                idx_train = np.random.randint(low=0, high=s, size=s)
                x_train = self.x_train.iloc[idx_train]
                y_train = self.y_train.iloc[idx_train]

            dict_params = {}


            pass