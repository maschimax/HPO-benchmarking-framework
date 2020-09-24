import numpy as np
import skopt
import time
from robo.fmin import bayesian_optimization, fabolas

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class RoboOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                         random_seed)

    def optimize(self) -> TuningResult:

        # Convert the skopt hyperparameter space into a continuous space for RoBO
        hp_space_lower = np.zeros(shape=(len(self.hp_space),))
        hp_space_upper = np.zeros(shape=(len(self.hp_space),))

        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                hp_space_lower[i,] = self.hp_space[i].low
                hp_space_upper[i,] = self.hp_space[i].high

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                n_choices = len(list(self.hp_space[i].categories))
                hp_space_lower[i,] = 0
                hp_space_upper[i,] = n_choices - 1

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                hp_space_lower[i,] = self.hp_space[i].low
                hp_space_upper[i,] = self.hp_space[i].high

            else:
                raise Exception('The skopt HP-space could not be converted correctly!')

        # Set the random seed of the random number generator
        rand_num_generator = np.random.RandomState(seed=self.random_seed)

        # Optimize on the predefined n_func_evals and measure the wall clock times
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Select the specified HPO-tuning method
        if self.hpo_method == 'Fabolas':
            # Budget correct? // Set further parameters?
            s_max = len(self.x_train)  # Maximum number of data points for the training data set
            s_min = int(0.05 * s_max)  # Maximum number of data points for the training data set
            n_init = int(self.n_func_evals / 3)  # Requirement of the fabolas implementation
            result_dict = fabolas(objective_function=self.objective_fabolas, s_min=s_min, s_max=s_max,
                                  lower=hp_space_lower, upper=hp_space_upper,
                                  num_iterations=self.n_func_evals, rng=rand_num_generator, n_init=n_init)

        elif self.hpo_method == 'Bohamiann':
            # Budget correct? // Set further parameters?
            result_dict = bayesian_optimization(objective_function=self.objective_bohamiann,
                                                lower=hp_space_lower, upper=hp_space_upper,
                                                model_type='bohamiann', num_iterations=self.n_func_evals,
                                                rng=rand_num_generator)

        else:
            raise Exception('Unknown HPO-method!')

        for i in range(len(self.times)):
            # Subtract the start time to receive the wall clock time of each function evaluation
            self.times[i] = self.times[i] - start_time
        wall_clock_time = max(self.times)

        if self.hpo_method == 'Fabolas':
            losses = result_dict['y']

        elif self.hpo_method == 'Bohamiann':
            losses = result_dict['incumbent_values']

        evaluation_ids = list(range(1, len(losses) + 1))
        best_loss = min(losses)

        configurations = ()
        for config in result_dict['incumbents']:
            config_dict = {}

            for i in range(len(config)):
                if type(self.hp_space[i]) == skopt.space.space.Integer:
                    config_dict[self.hp_space[i].name] = int(config[i])

                elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                    config_dict[self.hp_space[i].name] = list(self.hp_space[i].categories)[int(config[i])]

                elif type(self.hp_space[i]) == skopt.space.space.Real:
                    config_dict[self.hp_space[i].name] = config[i]

                else:
                    raise Exception('The continuous HP-space could not be converted correctly!')

            configurations = configurations + (config_dict,)

        best_configuration = configurations[-1]

        # Pass the results to a TuningResult-Object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=self.times, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_configuration,
                              wall_clock_time=wall_clock_time)

        return result

    def objective_fabolas(self, cont_hp_space, s):
        """

        :param self:
        :param cont_hp_space: np.array
        Array that contains the next hyperparameter configuration (continuous) to be evaluated
        :param s:
        Fraction of s_max
        :return:
        """

        # Transform the !continuous! hyperparameters into their respective types and save them in a dictionary
        dict_params = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                dict_params[self.hp_space[i].name] = int(cont_hp_space[i,])

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                dict_params[self.hp_space[i].name] = list(self.hp_space[i].categories)[int(cont_hp_space[i,])]

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                dict_params[self.hp_space[i].name] = cont_hp_space[i,]

            else:
                raise Exception('The continuous HP-space could not be converted correctly!')

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR':
            eval_func = self.train_evaluate_scikit_regressor

        elif self.ml_algorithm == 'KerasRegressor':
            eval_func = self.train_evaluate_keras_regressor

        elif self.ml_algorithm == 'XGBoostRegressor':
            eval_func = self.train_evaluate_xgboost_regressor

        else:
            raise Exception('Unknown ML-algorithm!')

        t_start_eval = time.time()
        val_loss = eval_func(params=dict_params, fabolas_budget=s)
        cost = time.time() - t_start_eval

        return val_loss, cost

    def objective_bohamiann(self, cont_hp_space):
        """

        :param cont_hp_space: np.array
        Array that contains the next hyperparameter configuration (continuous) to be evaluated
        :return:
        """

        # Transform the !continuous! hyperparameters into their respective types and save them in a dictionary
        dict_params = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                dict_params[self.hp_space[i].name] = int(cont_hp_space[i,])

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                dict_params[self.hp_space[i].name] = list(self.hp_space[i].categories)[int(cont_hp_space[i,])]

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                dict_params[self.hp_space[i].name] = cont_hp_space[i,]

            else:
                raise Exception('The continuous HP-space could not be converted correctly!')

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR':
            eval_func = self.train_evaluate_scikit_regressor

        elif self.ml_algorithm == 'KerasRegressor':
            eval_func = self.train_evaluate_keras_regressor

        elif self.ml_algorithm == 'XGBoostRegressor':
            eval_func = self.train_evaluate_xgboost_regressor

        else:
            raise Exception('Unknown ML-algorithm!')

        return eval_func(params=dict_params)
