from skopt.optimizer import forest_minimize, gp_minimize
import time

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class SkoptOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget,
                         random_seed)

    def optimize(self) -> TuningResult:

        # Select the specified HPO-tuning method
        if self.hpo_method == 'SMAC':
            this_optimizer = forest_minimize
            this_acq_func = 'EI'

        elif self.hpo_method == 'GPBO':
            this_optimizer = gp_minimize
            this_acq_func = 'EI'

        else:
            raise NameError('Unknown HPO-method!')

        # Optimize on the predefined budget and measure the wall clock times
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Start the optimization
        trial_result = this_optimizer(self.objective, self.hp_space, n_calls=self.budget, random_state=self.random_seed,
                                      acq_func=this_acq_func)

        for i in range(len(self.times)):
            # Subtract the start time to receive the wall clock time of each function evaluation
            self.times[i] = self.times[i] - start_time
        wall_clock_time = max(self.times)

        # Create a TuningResult-object to store the optimization results
        # Transformation of the results into a TuningResult-Object
        best_loss = trial_result.fun
        losses = list(trial_result.func_vals)  # Loss of each iteration

        # Determine the best HP-configuration of this run
        best_configuration = {}
        for i in range(len(self.hp_space)):
            best_configuration[self.hp_space[i].name] = trial_result.x[i]

        # Number the evaluations / iterations of this run
        evaluation_ids = list(range(1, len(trial_result.func_vals) + 1))

        # Determine the HP-configuration of each evaluation / iteration
        configurations = ()
        for i in range(len(trial_result.x_iters)):
            this_config = {}
            for j in range(len(self.hp_space)):
                this_config[self.hp_space[j].name] = trial_result.x_iters[i][j]
            configurations = configurations + (this_config,)

        # Pass the results to a TuningResult-object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=self.times, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_configuration,
                              wall_clock_time=wall_clock_time)

        return result

    def objective(self, params):
        """Objective function: This method converts the hyperparameters into a dictionary, passes them to the ML-model
         for training and returns the validation loss."""

        # Convert the hyperparameters into a dictionary to pass them to the ML-model
        dict_params = {}
        for i in range(len(self.hp_space)):
            dict_params[self.hp_space[i].name] = params[i]

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR':
            eval_func = self.train_evaluate_scikit_regressor

        elif self.ml_algorithm == 'KerasRegressor':
            eval_func = self.train_evaluate_keras_regressor

        elif self.ml_algorithm == 'XGBoostRegressor':
            eval_func = self.train_evaluate_xgboost_regressor

        else:
            raise NameError('Unknown ML-algorithm!')

        return eval_func(params=dict_params)
