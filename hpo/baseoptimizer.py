import pandas as pd
from abc import ABC, abstractmethod
import time
import functools

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow import keras
from xgboost import XGBRegressor

from hpo.results import TuningResult
from hpo.lr_schedules import fix, exponential, cosine


class BaseOptimizer(ABC):
    def __init__(self, hp_space, hpo_method: str, ml_algorithm: str,
                 x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
                 metric, budget: int, random_seed: int):
        """

        :param hp_space:
        :param hpo_method:
        :param ml_algorithm:
        :param x_train:
        :param x_val:
        :param y_train:
        :param y_val:
        :param metric
        :param budget:
        :param random_seed
        """

        self.hp_space = hp_space
        self.hpo_method = hpo_method
        self.ml_algorithm = ml_algorithm
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.metric = metric
        self.budget = budget
        self.random_seed = random_seed

    @abstractmethod
    def optimize(self) -> TuningResult:

        raise NotImplementedError

    def objective(self):

        raise NotImplementedError

    @staticmethod
    def get_best_configuration(result: TuningResult):
        # Returns the best configuration of this optimization run as a dictionary
        return result.best_configuration

    @staticmethod
    def get_best_score(result: TuningResult):
        # Returns the validation score of the best configuration of this optimization run
        raise result.best_loss

    @staticmethod
    def get_metrics(result: TuningResult):
        # Probably needs to be implemented in the Trial class
        raise NotImplementedError

    def train_evaluate_scikit_regressor(self, params: dict):
        """ This method trains a scikit-learn model according to the selected HP-configuration and returns the
        validation loss"""

        # Create ML-model for the HP-configuration selected by the HPO-method
        if self.ml_algorithm == 'RandomForestRegressor':
            model = RandomForestRegressor(**params, random_state=self.random_seed)
        elif self.ml_algorithm == 'SVR':
            model = SVR(**params)  # SVR has no random_state argument
        else:
            raise Exception('Unknown ML-algorithm!')

        # Train the model and make the prediction
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_val)

        # Compute the validation loss according to the metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss

    def train_evaluate_keras_regressor(self, params: dict):
        """ This method trains a keras model according to the selected HP-configuration and returns the
        validation loss"""

        # Initialize the neural network
        model = keras.Sequential()

        # Add input layer
        model.add(keras.layers.InputLayer(input_shape=len(self.x_train.keys())))

        # Add first hidden layer
        model.add(keras.layers.Dense(params['layer1_size'], activation=params['layer1_activation']))
        model.add(keras.layers.Dropout(params['dropout1']))

        # Add second hidden layer
        model.add(keras.layers.Dense(params['layer2_size'], activation=params['layer2_activation']))
        model.add(keras.layers.Dropout(params['dropout2']))

        # Add output layer
        model.add(keras.layers.Dense(1, activation='linear'))

        # Select optimizer and compile the model
        adam = keras.optimizers.Adam(learning_rate=params['init_lr'])
        model.compile(optimizer=adam, loss='mse', metrics=['mse'])

        # Learning rate schedule
        if params["lr_schedule"] == "cosine":
            schedule = functools.partial(cosine, initial_lr=params["init_lr"], T_max=self.budget)

        elif params["lr_schedule"] == "exponential":
            schedule = functools.partial(exponential, initial_lr=params["init_lr"], T_max=self.budget)

        elif params["lr_schedule"] == "constant":
            schedule = functools.partial(fix, initial_lr=params["init_lr"])

        else:
            raise Exception('Unknown learning rate schedule!')

        # Determine the learning rate for this iteration and pass it as callback
        lr = keras.callbacks.LearningRateScheduler(schedule)
        callbacks_list = [lr]

        # Train the model
        model.fit(self.x_train, self.y_train, epochs=self.budget, batch_size=params['batch_size'],
                  validation_data=(self.x_val, self.y_val), callbacks=callbacks_list,
                  verbose=1)

        # Make the prediction
        y_pred = model.predict(self.x_val)

        # Compute the validation loss according to the metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss

    def train_evaluate_xgboost_regressor(self, params: dict):
        """ This method trains a XGBoost model according to the selected HP-configuration and returns the
                validation loss"""

        # Initialize the model
        model = XGBRegressor(**params, random_state=self.random_seed)

        # Train the model and make the prediction
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_val)

        # Compute the validation loss according to the metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss