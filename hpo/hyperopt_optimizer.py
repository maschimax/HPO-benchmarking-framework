from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import skopt
import numpy as np
import time

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class HyperoptOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                         random_seed)

    def optimize(self) -> TuningResult:
        """
        Method performs a hyperparameter optimization run according to the selected HPO-method.
        :return: result: TuningResult
            TuningResult-object that contains the results of this optimization run.
        """

        # Select the specified HPO-tuning method
        if self.hpo_method == 'TPE':
            this_optimizer = tpe.suggest  # (seed=self.random_seed)

        else:
            raise Exception('Unknown HPO-method!')

        # Transform the skopt hp_space into an hyperopt space
        hyperopt_space = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                hyperopt_space[self.hp_space[i].name] = hp.choice(self.hp_space[i].name,
                                                                  range(self.hp_space[i].low,
                                                                        self.hp_space[i].high + 1))

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                hyperopt_space[self.hp_space[i].name] = hp.choice(self.hp_space[i].name,
                                                                  list(self.hp_space[i].categories))

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                hyperopt_space[self.hp_space[i].name] = hp.uniform(self.hp_space[i].name,
                                                                   low=self.hp_space[i].low,
                                                                   high=self.hp_space[i].high)

            else:
                raise Exception('The skopt HP-space could not be converted correctly!')

        # Initialize a trial instance
        trials = Trials()

        # Set the random seed of the random number generator
        rand_num_generator = np.random.RandomState(seed=self.random_seed)

        # Optimize on the predefined n_func_evals and measure the wall clock times
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Start the optimization
        res = fmin(fn=self.objective, space=hyperopt_space, trials=trials, algo=this_optimizer,
                   max_evals=self.n_func_evals, rstate=rand_num_generator)

        for i in range(len(self.times)):
            # Subtract the start time to receive the wall clock time of each function evaluation
            self.times[i] = self.times[i] - start_time
        wall_clock_time = max(self.times)

        # Create a TuningResult-object to store the optimization results
        # Transformation of the results into a TuningResult-Object

        # Number the evaluations / iterations of this run
        evaluation_ids = list(range(1, len(trials.tids) + 1))

        # Loss of each iteration
        losses = []
        for this_result in trials.results:
            losses.append(this_result['loss'])

        # Best loss
        best_loss = min(losses)

        # Determine the best HP-configuration of this run
        best_params = {}
        for i in range(len(self.hp_space)):

            if type(self.hp_space[i]) == skopt.space.space.Categorical:
                # Hyperopt only returns indexes for categorical hyperparameters
                categories = self.hp_space[i].categories
                cat_idx = res[self.hp_space[i].name]
                best_params[self.hp_space[i].name] = categories[cat_idx]

            else:
                best_params[self.hp_space[i].name] = res[self.hp_space[i].name]

        # HP-configuration of each iteration
        configurations = ()
        for trial in trials.trials:
            this_config = {}
            for i in range(len(self.hp_space)):

                if type(self.hp_space[i]) == skopt.space.space.Categorical:
                    # Hyperopt only returns indexes for categorical hyperparameters
                    categories = self.hp_space[i].categories
                    cat_idx = trial['misc']['vals'][self.hp_space[i].name][0]
                    this_config[self.hp_space[i].name] = categories[cat_idx]
                else:
                    this_config[self.hp_space[i].name] = trial['misc']['vals'][self.hp_space[i].name][0]

            configurations = configurations + (this_config,)

        # Pass the results to a TuningResult-object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=self.times, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_params,
                              wall_clock_time=wall_clock_time)

        return result

    def objective(self, params):
        """
        Objective function: This method passes the given hyperparameters to the ML-model for training and evaluation
        and returns the validation loss.
        :param params: dict
            Hyperparameter configuration that has been selected by the HPO-method for this iteration.
        :return: dict
            Dictionary that contains the validation loss, the optimization status and the evaluation time
        """
        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR' or \
                self.ml_algorithm == 'AdaBoostRegressor' or self.ml_algorithm == 'DecisionTreeRegressor':
            eval_func = self.train_evaluate_scikit_regressor

        elif self.ml_algorithm == 'KerasRegressor':
            eval_func = self.train_evaluate_keras_regressor

        elif self.ml_algorithm == 'XGBoostRegressor':
            eval_func = self.train_evaluate_xgboost_regressor

        else:
            raise Exception('Unknown ML-algorithm!')

        try:
            val_loss = eval_func(params=params)
            status = STATUS_OK
        except:
            status = STATUS_FAIL
            val_loss = float('nan')

        return {'loss': val_loss,
                'status': status,
                'eval_time': time.time()}
