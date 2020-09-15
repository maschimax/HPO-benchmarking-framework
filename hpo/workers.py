import time
import numpy as np
from math import sqrt

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow import keras


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
        rf_reg = RandomForestRegressor(random_state=0, n_jobs=-1,
                                       n_estimators=config['n_estimators'],
                                       max_depth=config['max_depth'], min_samples_leaf=config['min_samples_leaf'],
                                       min_samples_split=config['min_samples_split'],
                                       max_features=config['max_features'])

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
                 'info': {'validation_loss': val_loss}})

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
        svm_reg = SVR(C=config['C'], gamma=config['gamma'], epsilon=config['epsilon'])

        # Train the model on the specified budget
        n_train = len(self.X_train)
        n_fit = int(0.1 * budget * n_train)
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


# Keras Regressor
class KerasRegressor(Worker):
    def __init__(self, X_train, X_val, y_train, y_val, batch_size=32, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, *args, **kwargs):
        model = keras.Sequential()

        model.add(keras.layers.InputLayer(input_shape=[len(self.X_train.keys())]))

        model.add(keras.layers.Dense(config['width_1stlayer'], activation='relu'))

        if config['num_hidden_layers'] > 0:
            model.add(keras.layers.Dense(config['width_hidlayer1'], activation='relu'))

        if config['num_hidden_layers'] > 1:
            model.add(keras.layers.Dense(config['width_hidlayer2'], activation='relu'))

        if config['num_hidden_layers'] > 2:
            model.add(
                keras.layers.Dense(config['width_hidlayer3'], activation='relu'))

        if config['num_hidden_layers'] > 3:
            model.add(
                keras.layers.Dense(config['width_hidlayer4'], activation='relu'))

        model.add(keras.layers.Dropout(config['dropout_rate']))
        model.add(keras.layers.Dense(1))

        optimizer = keras.optimizers.RMSprop(learning_rate=config['lr'])

        model.compile(optimizer=optimizer, loss='mse')

        model.fit(self.X_train, self.y_train, epochs=int(budget), batch_size=self.batch_size,
                  validation_data=(self.X_val, self.y_val), verbose=0)

        y_pred = model.predict(self.X_val)
        val_loss = sqrt(mean_squared_error(self.y_val, y_pred))

        time.sleep(self.sleep_interval)

        return ({'loss': val_loss,
                 'info': {'validation_loss': val_loss}})

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        # HYPERPARAMETERS
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, log=True)

        num_hidden_layers = CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=1, upper=4, log=False)

        width_1stlayer = CSH.UniformIntegerHyperparameter('width_1stlayer', lower=8, upper=512, log=False)

        width_hidlayer1 = CSH.UniformIntegerHyperparameter('width_hidlayer1', lower=8, upper=256, log=False)

        width_hidlayer2 = CSH.UniformIntegerHyperparameter('width_hidlayer2', lower=8, upper=256, log=False)

        width_hidlayer3 = CSH.UniformIntegerHyperparameter('width_hidlayer3', lower=8, upper=256, log=False)

        width_hidlayer4 = CSH.UniformIntegerHyperparameter('width_hidlayer4', lower=8, upper=256, log=False)

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5,
                                                      log=False)

        cs.add_hyperparameters([lr, dropout_rate, num_hidden_layers, width_1stlayer, width_hidlayer1, width_hidlayer2,
                                width_hidlayer3, width_hidlayer4])

        return cs
