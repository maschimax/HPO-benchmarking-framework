import skopt
from skopt.optimizer import forest_minimize
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
import matplotlib.pyplot as plt

from hpo.optimizer import Optimizer


class Skopt_optimizer(Optimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget)

    def train_evaluate_rf_regressor(self, params):
        # How to deal with varying random states?
        rf_reg = RandomForestRegressor(**params, random_state=0)

        rf_reg.fit(self.x_train, self.y_train)
        y_pred = rf_reg.predict(self.x_val)

        val_loss = self.metric(self.y_val, y_pred)

        return val_loss

    def objective_rf_regressor(self, params):

        dict_params = {}
        for i in range(len(self.hp_space)):
            dict_params[self.hp_space[i].name] = params[i]

        return self.train_evaluate_rf_regressor(dict_params)

    def optimize(self):

        thisSpace = self.hp_space

        if self.ml_algorithm == 'RandomForestRegressor':
            thisObjective = self.objective_rf_regressor

        if self.hpo_method == 'SMAC':
            thisOptimizer = forest_minimize
            thisAcqFunction = 'EI'

        result = thisOptimizer(thisObjective, thisSpace, n_calls=self.budget, random_state=0, acq_func=thisAcqFunction)

        return result

    def get_best_configuration(self, result):
        # returns the best configuration as a dictionary
        dict_best_config = {}

        for i in range(len(self.hp_space)):
            dict_best_config[self.hp_space[i].name] = result.x[i]

        return dict_best_config

    def get_best_score(self, result):
        # returns the validation score of the best configuration
        return result.fun

    def plot_learning_curve(self, result):
        # Rework necessary
        loss_curve = result.func_vals
        best_loss_curve = []
        for i in range(len(loss_curve)):

            if i == 0:
                best_loss_curve.append(loss_curve[i])
            elif loss_curve[i] < min(best_loss_curve):
                best_loss_curve.append(loss_curve[i])
            else:
                best_loss_curve.append(min(best_loss_curve))

        fig, ax = plt.subplots()
        plt.plot(range(len(best_loss_curve)), best_loss_curve)
        plt.yscale('log')
        plt.xlabel('Number of evaluations')
        plt.ylabel('Loss')

        return plt.show()
