import optuna
import skopt
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from hpo.optimizer import Optimizer
from hpo.results import TuningResult


class Optuna_optimizer(Optimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget)

    def objective_rf_regressor(self, trial):
        params = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                params[self.hp_space[i].name] = trial.suggest_int(name=self.hp_space[i].name,
                                                                  low=self.hp_space[i].low,
                                                                  high=self.hp_space[i].high)

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                params[self.hp_space[i].name] = trial.suggest_categorical(name=self.hp_space[i].name,
                                                                          choices=list(self.hp_space[4].categories))
            # !! Add Remaining cases here
        rf_reg = RandomForestRegressor(random_state=0, **params)
        rf_reg.fit(self.x_train, self.y_train)
        y_pred = rf_reg.predict(self.x_val)

        val_loss = self.metric(self.y_val, y_pred)

        return val_loss

    def optimize(self):

        if self.ml_algorithm == 'RandomForestRegressor':
            thisObjective = self.objective_rf_regressor

        if self.hpo_method == 'TPE':
            thisOptimizer = TPESampler()

        study = optuna.create_study(sampler=thisOptimizer, direction='minimize')

        # Is the number of trials equal to the number of function evaluations?
        study.optimize(func=thisObjective, n_trials=self.budget)
        result = study.get_trials()
        best_params = study.best_params
        best_value = study.best_value

        result = TuningResult(best_value=best_value, best_configuration=best_params)
        return result

    def get_best_configuration(self, result):
        # returns the best configuration as a dictionary
        dict_best_config = {}
        return dict_best_config

    def plot_learning_curve(self, result):
        # Rework necessary
        best_loss_curve = []
        time_list = []
        for i in range(len(result)):
            if i == 0:
                best_loss_curve.append(result[i].value)
            elif result[i].value < min(best_loss_curve):
                best_loss_curve.append(result[i].value)
            else:
                best_loss_curve.append(min(best_loss_curve))

            time_list.append(result[i].datetime_complete)

        fig, ax = plt.subplots()
        plt.plot(time_list, best_loss_curve)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel('Loss')

        return plt.show()

