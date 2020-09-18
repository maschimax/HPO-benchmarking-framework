import optuna
import skopt
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class OptunaOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget, random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget, random_seed)

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

        return val_loss

    def optimize(self) -> TuningResult:

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor':
            thisObjective = self.objective_rf_regressor

        # Select the specified HPO-tuning method
        if self.hpo_method == 'TPE':
            thisOptimizer = TPESampler(seed=self.random_seed)

        # Create a study object and specify the optimization direction
        study = optuna.create_study(sampler=thisOptimizer, direction='minimize')

        # Optimize on the predefined budget
        # Is the number of trials equal to the number of function evaluations?
        study.optimize(func=thisObjective, n_trials=self.budget)

        # Create a TuningResult-object to store the optimization results
        all_trials = study.get_trials()
        best_params = study.best_params
        best_loss = study.best_value

        evaluation_ids = []
        timestamps = []
        losses = []
        configurations = ()
        for i in range(len(all_trials)):
            evaluation_ids.append(all_trials[i].number)
            timestamps.append(all_trials[i].datetime_complete)
            losses.append(all_trials[i].value)
            configurations = configurations + (all_trials[i].params,)

        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=timestamps, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_params)
        return result

    @staticmethod
    def plot_learning_curve(result: TuningResult):
        # Rework necessary
        fig, ax = plt.subplots()
        best_loss_curve = []
        loss_curve = result.losses
        for i in range(len(loss_curve)):
            if i == 0:
                best_loss_curve.append(loss_curve[i])
            elif loss_curve[i] < min(best_loss_curve):
                best_loss_curve.append(loss_curve[i])
            else:
                best_loss_curve.append(min(best_loss_curve))

        plt.plot(result.timestamps, best_loss_curve)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel('Loss')

        return plt.show()

