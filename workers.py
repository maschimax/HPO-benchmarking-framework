import time
import numpy as np
from math import sqrt

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Random Forest Classifier
class RandomForestWorker(Worker):
    def __init__(self, X_train, X_val, y_train, y_val, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, *args, **kwargs):
        rf_reg = RandomForestRegressor(random_state=0)

        # Train the model on the specified budget
        n_train = len(self.X_train)
        n_fit = int(0.1 * budget * n_train)
        ifit = np.random.randint(low=0, high=n_fit, size=n_fit)
        X_fit = self.X_train.iloc[ifit]
        y_fit = self.y_train.iloc[ifit]

        rf_reg.fit(X_fit, y_fit)

        y_pred = rf_reg.predict(self.X_val)
        val_loss = sqrt(mean_squared_error(self.y_val, y_pred))

        time.sleep(self.sleep_interval)

        return ({'loss': val_loss,
                 'info': {'validation_accuracy': val_loss}})

    # assign the configuration space to the worker by a static method
    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        # HYPERPARAMETERS
        n_estimators = CSH.UniformIntegerHyperparameter('n_estimators',
                                                        lower=10, upper=200)

        max_depth = CSH.UniformIntegerHyperparameter('max_depth',
                                                     lower=1, upper=80)

        min_samples_leaf = CSH.UniformIntegerHyperparameter('min_samples_leaf',
                                                            lower=1, upper=30)

        min_samples_split = CSH.UniformIntegerHyperparameter('min_samples_split',
                                                             lower=2, upper=20)

        max_features = CSH.CategoricalHyperparameter('max_features',
                                                     choices=['auto', 'sqrt'])

        # class_weight = CSH.CategoricalHyperparameter('class_weight',
        #    choices=['balanced', None])

        cs.add_hyperparameters([n_estimators, max_depth, min_samples_leaf,
                                min_samples_split, max_features])

        return cs


# SVM Classifier
class SVMWorker(Worker):
    def __init__(self, X_train, X_val, y_train, y_val, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, *args, **kwargs):
        svm_reg = SVR()

        # Train the model on the specified budget
        n_train = len(self.X_train)
        n_fit = int(0.1*budget*n_train)
        ifit = np.random.randint(low=0, high=n_fit, size=n_fit)
        X_fit = self.X_train.iloc[ifit]
        y_fit = self.y_train.iloc[ifit]

        svm_reg.fit(X_fit, y_fit)

        y_pred = svm_reg.predict(self.X_val)
        val_loss = sqrt(mean_squared_error(self.y_val, y_pred))

        time.sleep(self.sleep_interval)

        return ({'loss': val_loss,
                 'info': {'validation_loss': val_loss}})

    # assign the configuration space to the worker by a static method
    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        # HYPERPARAMETERS
        C = CSH.UniformFloatHyperparameter('C', lower=1e-3, upper=1e+3)

        gamma = CSH.CategoricalHyperparameter('gamma', choices=['scale', 'auto'])

        epsilon = CSH.UniformFloatHyperparameter('epsilon', lower=1e-3,
                                                 upper=1e+0)

        cs.add_hyperparameters([C, gamma, epsilon])

        return cs
