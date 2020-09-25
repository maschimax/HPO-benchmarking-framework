import optuna
import skopt
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
import time

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class OptunaOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                         random_seed)

    def optimize(self) -> TuningResult:

        # Select the specified HPO-tuning method
        if self.hpo_method == 'TPE':
            this_optimizer = TPESampler(seed=self.random_seed)

        elif self.hpo_method == 'CMA-ES':
            this_optimizer = CmaEsSampler(seed=self.random_seed)

        elif self.hpo_method == 'RandomSearch':
            this_optimizer = RandomSampler(seed=self.random_seed)

        else:
            raise Exception('Unknown HPO-method!')

        # Create a study object and specify the optimization direction
        study = optuna.create_study(sampler=this_optimizer, direction='minimize')

        # Optimize on the predefined n_func_evals and measure the wall clock times
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Start the optimization
        study.optimize(func=self.objective, n_trials=self.n_func_evals)

        for i in range(len(self.times)):
            # Subtract the start time to receive the wall clock time of each function evaluation
            self.times[i] = self.times[i] - start_time
        wall_clock_time = max(self.times)

        # Create a TuningResult-object to store the optimization results
        # Transformation of the results into a TuningResult-Object
        all_trials = study.get_trials()
        best_params = study.best_params
        best_loss = study.best_value

        evaluation_ids = []  # Number the evaluations / iterations of this run
        losses = []  # Loss of each iteration
        configurations = ()  # HP-configuration of each iteration
        for i in range(len(all_trials)):
            evaluation_ids.append(all_trials[i].number)
            losses.append(all_trials[i].value)
            configurations = configurations + (all_trials[i].params,)

        # Pass the results to a TuningResult-object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=self.times, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_params,
                              wall_clock_time=wall_clock_time)

        return result

    def objective(self, trial):
        """
        Objective function: This method converts the hyperparameters into a dictionary, passes them to the ML-model
        for training and returns the validation loss.
        :param trial:
        :return:
        """

        # Convert the hyperparameters into a dictionary to pass them to the ML-model
        dict_params = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                dict_params[self.hp_space[i].name] = trial.suggest_int(name=self.hp_space[i].name,
                                                                       low=self.hp_space[i].low,
                                                                       high=self.hp_space[i].high)

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                dict_params[self.hp_space[i].name] = trial.suggest_categorical(name=self.hp_space[i].name,
                                                                               choices=list(self.hp_space[i].categories))

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                dict_params[self.hp_space[i].name] = trial.suggest_float(name=self.hp_space[i].name,
                                                                         low=self.hp_space[i].low,
                                                                         high=self.hp_space[i].high)
            else:
                raise Exception('The skopt HP-space could not be converted correctly!')

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
