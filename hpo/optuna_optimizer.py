import optuna
import skopt
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
import time

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class OptunaOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget,
                         random_seed)

    def optimize(self) -> TuningResult:

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor':
            this_objective = self.objective_rf_regressor

        else:
            raise NameError('Unknown ML-algorithm!')

        # Select the specified HPO-tuning method
        if self.hpo_method == 'TPE':
            this_optimizer = TPESampler(seed=self.random_seed)

        else:
            raise NameError('Unknown HPO-method!')

        # Create a study object and specify the optimization direction
        study = optuna.create_study(sampler=this_optimizer, direction='minimize')

        # Optimize on the predefined budget and measure the wall clock times
        # Is the number of trials equal to the number of function evaluations?
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Start the optimization
        study.optimize(func=this_objective, n_trials=self.budget)

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

    def objective_rf_regressor(self, trial):
        # Objective function for a RandomForestRegressor

        # Convert the skopt HP-space into an optuna HP-space
        params = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                params[self.hp_space[i].name] = trial.suggest_int(name=self.hp_space[i].name,
                                                                  low=self.hp_space[i].low,
                                                                  high=self.hp_space[i].high)

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                params[self.hp_space[i].name] = trial.suggest_categorical(name=self.hp_space[i].name,
                                                                          choices=list(self.hp_space[4].categories))

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                params[self.hp_space[i].name] = trial.suggest_float(name=self.hp_space[i].name,
                                                                    low=self.hp_space[i].low,
                                                                    high=self.hp_space[i].high)
            else:
                raise NameError('The skopt HP-space could not be converted correctly!')

        # Create ML-model for the HP-configuration selected by the HPO-method
        rf_reg = RandomForestRegressor(random_state=self.random_seed, **params)
        rf_reg.fit(self.x_train, self.y_train)
        y_pred = rf_reg.predict(self.x_val)

        # Compute the validation loss according to the metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss
