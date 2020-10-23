import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import time
import functools

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from tensorflow import keras
from xgboost import XGBRegressor

from hpo_framework.results import TuningResult
from hpo_framework.lr_schedules import fix, exponential, cosine


class BaseOptimizer(ABC):
    def __init__(self, hp_space, hpo_method: str, ml_algorithm: str,
                 x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
                 metric, n_func_evals: int, random_seed: int, n_workers: int):
        """
        Superclass for the individual optimizer classes of each HPO-library.
        :param hp_space:
        :param hpo_method:
        :param ml_algorithm:
        :param x_train:
        :param x_val:
        :param y_train:
        :param y_val:
        :param metric
        :param n_func_evals:
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
        self.n_func_evals = n_func_evals
        self.random_seed = random_seed
        self.n_workers = n_workers

    @abstractmethod
    def optimize(self) -> TuningResult:

        raise NotImplementedError

    @staticmethod
    def get_best_configuration(result: TuningResult):
        """
        Method returns the best hyperparameter configuration of this optimization run.
        :param result: TuningResult
            TuningResult-object that contains the results of an optimization run.
        :return: result.best_configuration: dict
            Best hp-configuration.
        """
        return result.best_configuration

    @staticmethod
    def get_best_score(result: TuningResult):
        """
        Method returns the best best loss value of this optimization run.
        :param result: TuningResult
            TuningResult-object that contains the results of an optimization run.
        :return: result.best_loss: dict
            Best achieved loss of this optimization run.
        """
        # Returns the validation score of the best configuration of this optimization run
        raise result.best_loss

    def impute_results_for_crash(self):
        """
        In case the optimization fails, this method generates default values for the variables that are expected as the
        result of an optimization run.
        :return:
            Imputed values for the tuning results variables.
        """
        evaluation_ids = [float('nan')] * self.n_func_evals
        timestamps = [float('nan')] * self.n_func_evals
        losses = [float('nan')] * self.n_func_evals
        configurations = tuple([float('nan')] * self.n_func_evals)
        best_loss = [float('nan')]
        best_configuration = {'params': None}
        wall_clock_time = float('nan')
        return evaluation_ids, timestamps, losses, configurations, best_loss, best_configuration, wall_clock_time

    def get_warmstart_configuration(self):
        """
        Determine the default hyperparameter configuration of the selected ML-algorithm. This configuration can be used
        as a warmstart configuration for the HPO-method.
        :return: default_params: dict
            Dictionary that contains the default HPs.
        """
        if self.ml_algorithm == 'RandomForestRegressor':
            default_model = RandomForestRegressor(random_state=self.random_seed)

        elif self.ml_algorithm == 'SVR':
            # SVR has no random_state parameter
            default_model = SVR()

        elif self.ml_algorithm == 'AdaBoostRegressor':
            default_model = AdaBoostRegressor(random_state=self.random_seed)

        elif self.ml_algorithm == 'DecisionTreeRegressor':
            default_model = DecisionTreeRegressor(random_state=self.random_seed)

        elif self.ml_algorithm == 'LinearRegression':
            # LinearRegression has no random_state parameter
            default_model = LinearRegression()

        elif self.ml_algorithm == 'KNNRegressor':
            # KNeighborsRegressor has no random_state parameter
            default_model = KNeighborsRegressor()

            # Add remaining ML-algorithms here (e.g. XGBoost, Keras)

        else:
            raise Exception('Unknown ML-algorithm!')

        # Default HPs of the ML-algorithm
        default_params = default_model.get_params()

        return default_params

    def get_warmstart_loss(self, **kwargs):
        """
        Computes the loss of the selected ML-algorithm for the default hyperparameter configuration or any valid
         configuration that has been passed via kwargs.
        :param kwargs: dict
            Possibility to pass any valid HP-configuration for the ML-algorithm. If a argument 'warmstart_dict' is
             passed, this configuration is used to compute the loss.
        :return: warmstart_loss: float
            Validation loss for the default HP-configuration or the HP-configuration that has been passed via kwargs.
        """

        # Check, whether a warmstart configuration was passed
        if 'warmstart_dict' in kwargs:
            warmstart_config = kwargs['warmstart_dict']

        # Otherwise use the default parameters of the ML-algorithm
        else:
            warmstart_config = self.get_warmstart_configuration()

        # Use the warmstart HP-configuration to create a model for the ML-algorithm selected
        if self.ml_algorithm == 'RandomForestRegressor':
            model = RandomForestRegressor(**warmstart_config, random_state=self.random_seed)

        elif self.ml_algorithm == 'SVR':
            # SVR has no random_state parameter
            model = SVR(**warmstart_config)

        elif self.ml_algorithm == 'AdaBoostRegressor':
            model = AdaBoostRegressor(**warmstart_config, random_state=self.random_seed)

        elif self.ml_algorithm == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(**warmstart_config, random_state=self.random_seed)

        elif self.ml_algorithm == 'LinearRegression':
            # LinearRegression has no random_state parameter
            model = LinearRegression(**warmstart_config)

        elif self.ml_algorithm == 'KNNRegressor':
            # KNeighborsRegressor has no random_state parameter
            model = KNeighborsRegressor(**warmstart_config)

            # Add remaining ML-algorithms here (e.g. XGBoost, Keras)

        else:
            raise Exception('Unknown ML-algorithm!')

        # Train the model and make the prediction
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_val)

        # Compute the warmstart (validation) loss according to the loss_metric selected
        warmstart_loss = self.metric(self.y_val, y_pred)

        return warmstart_loss

    def train_evaluate_scikit_model(self, params: dict, **kwargs):
        """
        This method trains a scikit-learn model according to the selected HP-configuration and returns the
        validation loss
        :param params: dict
            Dictionary of hyperparameters
        :param kwargs: dict
            Further keyword arguments (e.g. hp_budget: share of training set (x_train, y_train))
        :return: val_loss: float
            Validation loss of this run
        """

        # Create ML-model for the HP-configuration selected by the HPO-method
        if self.ml_algorithm == 'RandomForestRegressor':
            model = RandomForestRegressor(**params, random_state=self.random_seed, n_jobs=self.n_workers)

        elif self.ml_algorithm == 'RandomForestClassifier':
            model = RandomForestClassifier(**params, random_state=self.random_seed, n_jobs=self.n_workers)

        elif self.ml_algorithm == 'SVR':
            # SVR has no random_state and no n_jobs parameters
            model = SVR(**params)

        elif self.ml_algorithm == 'AdaBoostRegressor':
            # AdaBoostRegrssor has no n_jobs parameter
            model = AdaBoostRegressor(**params, random_state=self.random_seed)

        elif self.ml_algorithm == 'DecisionTreeRegressor':
            # DecisionTreeRegressor has no n_jobs parameter
            model = DecisionTreeRegressor(**params, random_state=self.random_seed)

        elif self.ml_algorithm == 'LinearRegression':
            # LinearRegression has no random_state parameter
            model = LinearRegression(**params, n_jobs=self.n_workers)

        elif self.ml_algorithm == 'KNNRegressor':
            # KNeighborsRegressor has no random_state parameter
            model = KNeighborsRegressor(**params, n_jobs=self.n_workers)

        else:
            raise Exception('Unknown ML-algorithm!')

        if 'hb_budget' in kwargs:
            # For BOHB and Hyperband select the training data according to the budget of this iteration
            hb_budget = kwargs['hb_budget']
            n_train = len(self.x_train)
            n_budget = int(0.1 * hb_budget * n_train)
            idx_train = np.random.randint(low=0, high=n_budget, size=n_budget)
            x_train = self.x_train.iloc[idx_train]
            y_train = self.y_train.iloc[idx_train]

        elif 'fabolas_budget' in kwargs:
            # For Fabolas select the training data according to the budget of this iteration
            fabolas_budget = kwargs['fabolas_budget']
            idx_train = np.random.randint(low=0, high=fabolas_budget, size=fabolas_budget)
            x_train = self.x_train.iloc[idx_train]
            y_train = self.y_train.iloc[idx_train]

        else:
            x_train = self.x_train
            y_train = self.y_train

        # Train the model and make the prediction
        model.fit(x_train, y_train)
        y_pred = model.predict(self.x_val)

        # Compute the validation loss according to the loss_metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss

    def train_evaluate_keras_regressor(self, params: dict, **kwargs):
        """
        This method trains a keras model according to the selected HP-configuration and returns the
        validation loss
        :param params: dict
            Dictionary of hyperparameters
        :param kwargs: dict
            Further keyword arguments (e.g. hp_budget: share of the total number of epochs for training)
        :return: val_loss: float
            Validation loss of this run
        """
        full_budget_epochs = 100  # see https://arxiv.org/abs/1905.04970

        if 'hb_budget' in kwargs:
            # For BOHB and Hyperband select the number of epochs according to the budget of this iteration
            hb_budget = kwargs['hb_budget']
            epochs = int(0.1 * hb_budget * full_budget_epochs)
            x_train = self.x_train
            y_train = self.y_train

        elif 'fabolas_budget' in kwargs:
            # For Fabolas select the training data according to the budget of this iteration
            fabolas_budget = kwargs['fabolas_budget']
            idx_train = np.random.randint(low=0, high=fabolas_budget, size=fabolas_budget)
            x_train = self.x_train.iloc[idx_train]
            y_train = self.y_train.iloc[idx_train]
            epochs = full_budget_epochs

        else:
            x_train = self.x_train
            y_train = self.y_train
            epochs = full_budget_epochs  # train on the full budget

        # Initialize the neural network
        model = keras.Sequential()

        # Add input layer
        model.add(keras.layers.InputLayer(input_shape=len(x_train.keys())))

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
            schedule = functools.partial(cosine, initial_lr=params["init_lr"], T_max=epochs)

        elif params["lr_schedule"] == "exponential":
            schedule = functools.partial(exponential, initial_lr=params["init_lr"], T_max=epochs)

        elif params["lr_schedule"] == "constant":
            schedule = functools.partial(fix, initial_lr=params["init_lr"])

        else:
            raise Exception('Unknown learning rate schedule!')

        # Determine the learning rate for this iteration and pass it as callback
        lr = keras.callbacks.LearningRateScheduler(schedule)
        callbacks_list = [lr]

        # Train the model
        model.fit(x_train, y_train, epochs=epochs, batch_size=params['batch_size'],
                  validation_data=(self.x_val, self.y_val), callbacks=callbacks_list,
                  verbose=1)

        # Make the prediction
        y_pred = model.predict(self.x_val)

        # Compute the validation loss according to the loss_metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss

    def train_evaluate_xgboost_regressor(self, params: dict, **kwargs):
        """
        This method trains a XGBoost model according to the selected HP-configuration and returns the
        validation loss
        :param params: dict
            Dictionary of hyperparameters
        :param kwargs: dict
            Further keyword arguments (e.g. hp_budget: share of training set (x_train, y_train))
        :return: val_loss: float
            Validation loss of this run
        """

        if 'hb_budget' in kwargs:
            # For BOHB and Hyperband select the training data according to the budget of this iteration
            hb_budget = kwargs['hb_budget']
            n_train = len(self.x_train)
            n_budget = int(0.1 * hb_budget * n_train)
            idx_train = np.random.randint(low=0, high=n_budget, size=n_budget)
            x_train = self.x_train.iloc[idx_train]
            y_train = self.y_train.iloc[idx_train]

        elif 'fabolas_budget' in kwargs:
            # For Fabolas select the training data according to the budget of this iteration
            fabolas_budget = kwargs['fabolas_budget']
            idx_train = np.random.randint(low=0, high=fabolas_budget, size=fabolas_budget)
            x_train = self.x_train.iloc[idx_train]
            y_train = self.y_train.iloc[idx_train]

        else:
            x_train = self.x_train
            y_train = self.y_train

        # Initialize the model
        model = XGBRegressor(**params, random_state=self.random_seed, n_jobs=self.n_workers)

        # Train the model and make the prediction
        model.fit(x_train, y_train)
        y_pred = model.predict(self.x_val)

        # Compute the validation loss according to the loss_metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss
