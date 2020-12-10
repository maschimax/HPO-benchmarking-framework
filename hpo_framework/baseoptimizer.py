import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import time
import functools

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import KFold, train_test_split
from tensorflow import keras
from tensorflow.random import set_seed
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb

from hpo_framework.results import TuningResult
from hpo_framework.lr_schedules import fix, exponential, cosine
from hpo_framework.hp_spaces import warmstart_lgb, warmstart_xgb, warmstart_keras, warmstart_dt, warmstart_knn, \
    warmstart_svm, warmstart_rf_clf, warmstart_rf_reg, warmstart_ada_clf, warmstart_ada_reg


class BaseOptimizer(ABC):
    def __init__(self, hp_space, hpo_method: str, ml_algorithm: str,
                 x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                 metric, n_func_evals: int, random_seed: int, n_workers: int, cross_val: bool):
        """
        Superclass for the individual optimizer classes of each HPO-library.
        :param hp_space: list
            List that stores the hyperparameter space in scikit optimize format.
        :param hpo_method: str
            Specifies the HPO method to use for hyperparameter optimization.
        :param ml_algorithm: str
            Specifies the ML algorithm.
        :param x_train: pd.DataFrame
            Training data set (features only).
        :param x_test: pd.DataFrame
            Test data set (features only).
        :param y_train: pd.Series
            Labels of the training data.
        :param y_test: pd.Series
            Labels of the test data.
        :param metric
            Reference to the loss metric function.
        :param n_func_evals: int
            Number of blackbox function evaluations that is allowed during the optimization (budget).
        :param random_seed: int
            Seed for random number generator
        """

        self.hp_space = hp_space
        self.hpo_method = hpo_method
        self.ml_algorithm = ml_algorithm
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.metric = metric
        self.n_func_evals = n_func_evals
        self.random_seed = random_seed
        self.n_workers = n_workers
        self.cross_val = cross_val

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

    def get_best_val_score(result: TuningResult):
        """
        Method returns the best best validation loss  of this optimization run.
        :param result: TuningResult
            TuningResult-object that contains the results of an optimization run.
        :return: result.best_loss: dict
            Best achieved validation loss of this optimization run.
        """
        # Returns the validation score of the best configuration of this optimization run
        raise result.best_val_loss

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
        best_loss = float('nan')
        best_configuration = {'params': None}
        wall_clock_time = float('nan')
        test_loss = float('nan')
        budget = [float('nan')] * self.n_func_evals

        return evaluation_ids, timestamps, losses, configurations, best_loss, best_configuration, wall_clock_time, \
               test_loss, budget

    def get_warmstart_configuration(self):
        """
        Return the warmstart hyperparameter configuration of the selected ML-algorithm. This configuration contains either
        promising values from benchmarks / studies or the default value of the algorithm's implementation.
        :return: warmstart_params: dict
            Dictionary that contains the warmstart HPs.
        """
        if self.ml_algorithm == 'MLPRegressor':
            default_model = MLPRegressor(random_state=self.random_seed)
            warmstart_params = default_model.get_params()

        elif self.ml_algorithm == 'MLPClassifier':
            default_model = MLPClassifier(random_state=self.random_seed)
            warmstart_params = default_model.get_params()

        elif self.ml_algorithm == 'RandomForestRegressor':
            warmstart_params = warmstart_rf_reg
            warmstart_params['random_state'] = self.random_seed

        elif self.ml_algorithm == 'RandomForestClassifier':
            warmstart_params = warmstart_rf_clf
            warmstart_params['random_state'] = self.random_seed

        elif self.ml_algorithm == 'SVR':
            # SVR has no random_state parameter
            warmstart_params = warmstart_svm

        elif self.ml_algorithm == 'SVC':
            warmstart_params = warmstart_svm
            warmstart_params['random_state'] = self.random_seed

        elif self.ml_algorithm == 'AdaBoostRegressor':
            warmstart_params = warmstart_ada_reg
            warmstart_params['random_state'] = self.random_seed

        elif self.ml_algorithm == 'AdaBoostClassifier':
            warmstart_params = warmstart_ada_clf
            warmstart_params['random_state'] = self.random_seed

        elif self.ml_algorithm == 'DecisionTreeRegressor' or self.ml_algorithm == 'DecisionTreeClassifier':
            warmstart_params = warmstart_dt
            warmstart_params['random_state'] = self.random_seed

        elif self.ml_algorithm == 'LinearRegression':
            # LinearRegression has no random_state parameter
            default_model = LinearRegression()
            warmstart_params = default_model.get_params()

        elif self.ml_algorithm == 'LogisticRegression':
            default_model = LogisticRegression(random_state=self.random_seed)
            warmstart_params = default_model.get_params()

        elif self.ml_algorithm == 'KNNRegressor' or self.ml_algorithm == 'KNNClassifier':
            # KNeighborsRegressor has no random_state parameter
            warmstart_params = warmstart_knn
            warmstart_params['random_state'] = self.random_seed

        elif self.ml_algorithm == 'NaiveBayes':
            # GaussianNB has no random_state parameter
            default_model = GaussianNB()
            warmstart_params = default_model.get_params()

        elif self.ml_algorithm == 'ElasticNet':
            default_model = ElasticNet()
            warmstart_params = default_model.get_params()

        elif self.ml_algorithm == 'KerasRegressor' or self.ml_algorithm == 'KerasClassifier':
            warmstart_params = warmstart_keras

        elif self.ml_algorithm == 'XGBoostRegressor' or self.ml_algorithm == 'XGBoostClassifier':
            warmstart_params = warmstart_xgb
            warmstart_params['random_state'] = self.random_seed

        elif self.ml_algorithm == 'LGBMRegressor' or self.ml_algorithm == 'LGBMClassifier':
            warmstart_params = warmstart_lgb
            warmstart_params['seed'] = self.random_seed

            # Add remaining ML-algorithms here

        else:
            raise Exception('Unknown ML-algorithm!')

        # Return the default HPs of the ML-algorithm
        return warmstart_params

    def get_warmstart_loss(self, cv_mode=True, **kwargs):
        """
        Computes the validation loss of the selected ML-algorithm for the warmstart hyperparameter configuration or any
        valid configuration that has been passed via kwargs
        :param cv_mode: bool
            Flag that indicates, whether to perform cross validation or simple validation
        :param kwargs: dict
            Possibility to pass any valid HP-configuration for the ML-algorithm. If a argument 'warmstart_dict' is
             passed, this configuration is used to compute the loss.
        :return: warmstart_loss: float
            Validation loss for the default HP-configuration or the HP-configuration that has been passed via kwargs.
        """
        # Create K-Folds cross validator
        kf = KFold(n_splits=5)
        cross_val_losses = []
        cv_iter = 0

        # Iterate over the cross validation splits
        for train_index, val_index in kf.split(X=self.x_train):

            # Cross validation
            if cv_mode:

                x_train_cv, x_val_cv = self.x_train.iloc[train_index], self.x_train.iloc[val_index]
                y_train_cv, y_val_cv = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            # Separate a validation set, but do not perform cross validation
            elif not cv_mode and cv_iter < 2:

                x_train_cv, x_val_cv, y_train_cv, y_val_cv = train_test_split(self.x_train, self.y_train, test_size=0.2,
                                                                              shuffle=True, random_state=0)
            # Iteration doesn't make sense for non cross validation
            else:
                continue

            # Check, whether a warmstart configuration was passed
            if 'warmstart_dict' in kwargs:
                warmstart_config = kwargs['warmstart_dict']

            # Otherwise use the default parameters of the ML-algorithm
            else:
                warmstart_config = self.get_warmstart_configuration()

            # Use the warmstart HP-configuration to create a model for the ML-algorithm selected
            if self.ml_algorithm == 'MLPRegressor':
                model = MLPRegressor(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'MLPClassifier':
                model = MLPClassifier(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'RandomForestRegressor':
                model = RandomForestRegressor(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'RandomForestClassifier':
                model = RandomForestClassifier(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'SVR':
                warmstart_config['cache_size'] = 500
                model = SVR(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'SVC':
                warmstart_config['cache_size'] = 500
                model = SVC(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'AdaBoostRegressor':
                model = AdaBoostRegressor(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'AdaBoostClassifier':
                model = AdaBoostClassifier(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'DecisionTreeRegressor':
                model = DecisionTreeRegressor(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'DecisionTreeClassifier':
                model = DecisionTreeClassifier(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'LinearRegression':
                model = LinearRegression(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'KNNRegressor':
                model = KNeighborsRegressor(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'KNNClassifier':
                model = KNeighborsClassifier(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'LogisticRegression':
                model = LogisticRegression(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'NaiveBayes':
                model = GaussianNB(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'ElasticNet':
                model = ElasticNet(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'KerasRegressor' or self.ml_algorithm == 'KerasClassifier':

                epochs = 100

                # Set the (global) random seed for TensorFlow
                set_seed(self.random_seed)

                # Initialize the neural network
                model = keras.Sequential()

                # Add input layer
                model.add(keras.layers.InputLayer(input_shape=len(x_train_cv.keys())))

                # Add first hidden layer
                if warmstart_config['hidden_layer1_size'] > 0:
                    model.add(
                        keras.layers.Dense(warmstart_config['hidden_layer1_size'],
                                           activation=warmstart_config['hidden_layer1_activation']))
                    model.add(keras.layers.Dropout(warmstart_config['dropout1']))

                # Add second hidden layer
                if warmstart_config['hidden_layer2_size'] > 0:
                    model.add(
                        keras.layers.Dense(warmstart_config['hidden_layer2_size'],
                                           activation=warmstart_config['hidden_layer2_activation']))
                    model.add(keras.layers.Dropout(warmstart_config['dropout2']))

                # Add third hidden layer
                if warmstart_config['hidden_layer3_size'] > 0:
                    model.add(
                        keras.layers.Dense(warmstart_config['hidden_layer3_size'],
                                           activation=warmstart_config['hidden_layer3_activation']))
                    model.add(keras.layers.Dropout(warmstart_config['dropout3']))

                # Add fourth hidden layer
                if warmstart_config['hidden_layer4_size'] > 0:
                    model.add(
                        keras.layers.Dense(warmstart_config['hidden_layer4_size'],
                                           activation=warmstart_config['hidden_layer4_activation']))
                    model.add(keras.layers.Dropout(warmstart_config['dropout4']))

                # Add output layer
                if self.ml_algorithm == 'KerasRegressor':

                    model.add(keras.layers.Dense(1, activation='linear'))

                    # Select optimizer and compile the model
                    adam = keras.optimizers.Adam(learning_rate=warmstart_config['init_lr'])
                    model.compile(optimizer=adam, loss='mse', metrics=['mse'])

                elif self.ml_algorithm == 'KerasClassifier':

                    # Determine the number of different classes depending on the data format
                    if type(y_train_cv) == pd.core.series.Series:
                        num_classes = int(max(y_train_cv) - min(y_train_cv) + 1)

                    elif type(y_train_cv) == pd.core.frame.DataFrame:
                        num_classes = len(y_train_cv.keys())

                    else:
                        raise Exception('Unknown data format!')

                    # Binary classification
                    if num_classes <= 2:

                        # 'Sigmoid is equivalent to a 2-element Softmax, where the second element is assumed to be zero'
                        # https://keras.io/api/layers/activations/#sigmoid-function
                        model.add(keras.layers.Dense(1, activation='sigmoid'))

                        adam = keras.optimizers.Adam(learning_rate=warmstart_config['init_lr'])
                        model.compile(optimizer=adam, loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

                    # Multiclass classification
                    else:

                        # Use softmax activation for multiclass clf. -> 'Softmax converts a real vector to a vector of
                        # categorical probabilities.[...]the result could be interpreted as a probability distribution.'
                        # https://keras.io/api/layers/activations/#softmax-function
                        model.add(keras.layers.Dense(num_classes, activation='softmax'))

                        adam = keras.optimizers.Adam(learning_rate=warmstart_keras['init_lr'])
                        model.compile(optimizer=adam, loss=keras.losses.CategoricalCrossentropy(),
                                      metrics=[keras.metrics.CategoricalAccuracy()])

                # Learning rate schedule
                if warmstart_config["lr_schedule"] == "cosine":
                    schedule = functools.partial(cosine, initial_lr=warmstart_config["init_lr"], T_max=epochs)

                elif warmstart_config["lr_schedule"] == "exponential":
                    schedule = functools.partial(exponential, initial_lr=warmstart_config["init_lr"], T_max=epochs)

                elif warmstart_config["lr_schedule"] == "constant":
                    schedule = functools.partial(fix, initial_lr=warmstart_config["init_lr"])

                else:
                    raise Exception('Unknown learning rate schedule!')

                # Determine the learning rate for this iteration and pass it as callback
                lr = keras.callbacks.LearningRateScheduler(schedule)

                # Early stopping callback
                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               min_delta=0,
                                                               patience=10,
                                                               verbose=1,
                                                               mode='auto',
                                                               restore_best_weights=True)

                callbacks_list = [lr, early_stopping]

                # Train the model
                model.fit(x_train_cv, y_train_cv, epochs=epochs, batch_size=warmstart_config['batch_size'],
                          validation_data=(x_val_cv, y_val_cv), callbacks=callbacks_list,
                          verbose=0)

                # Make the prediction
                y_pred = model.predict(x_val_cv)

                # In case of binary classification round to the neares integer
                if self.ml_algorithm == 'KerasClassifier':

                    # Binary classification
                    if num_classes <= 2:

                        y_pred = np.rint(y_pred)

                    # Multiclass classification
                    else:

                        # Identify the predicted class (maximum probability) in each row
                        for row_idx in range(y_pred.shape[0]):

                            # Predicted class
                            this_class = np.argmax(y_pred[row_idx, :])

                            # Iterate over columns / classes
                            for col_idx in range(y_pred.shape[1]):

                                if col_idx == this_class:
                                    y_pred[row_idx, col_idx] = 1
                                else:
                                    y_pred[row_idx, col_idx] = 0

                # KerasRegressor
                else:
                    y_pred = np.reshape(y_pred, newshape=(-1,))

            elif self.ml_algorithm == 'XGBoostRegressor' or self.ml_algorithm == 'XGBoostClassifier':

                # Consideration of conditional hyperparameters
                if warmstart_config['booster'] not in ['gbtree', 'dart']:
                    del warmstart_config['eta']
                    del warmstart_config['subsample']
                    del warmstart_config['max_depth']
                    del warmstart_config['min_child_weight']
                    del warmstart_config['colsample_bytree']
                    del warmstart_config['colsample_bylevel']

                if self.ml_algorithm == 'XGBoostRegressor':

                    model = XGBRegressor(**warmstart_config)

                elif self.ml_algorithm == 'XGBoostClassifier':

                    model = XGBClassifier(**warmstart_config)

                # Train the model and make the prediction
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'LGBMRegressor' or self.ml_algorithm == 'LGBMClassifier':
                train_data = lgb.Dataset(x_train_cv, y_train_cv)
                valid_data = lgb.Dataset(x_val_cv, y_val_cv)

                if 'objective' not in warmstart_config.keys():
                    # Specify the ML task
                    if self.ml_algorithm == 'LGBMRegressor':
                        # Regression task
                        warmstart_config['objective'] = 'regression'

                    elif self.ml_algorithm == 'LGBMClassifier':

                        # Determine the number of classes
                        num_classes = int(max(y_train_cv) - min(y_train_cv) + 1)

                        # Binary classification task
                        if num_classes <= 2:
                            warmstart_config['objective'] = 'binary'

                        # Multiclass classification task
                        else:
                            warmstart_config['objective'] = 'multiclass'  # uses Softmax objective function
                            warmstart_config['num_class'] = num_classes

                if 'seed' not in warmstart_config.keys():
                    # Specify the random seed
                    warmstart_config['seed'] = self.random_seed

                # Train the model and make the prediction
                model = lgb.train(params=warmstart_config, train_set=train_data, valid_sets=[valid_data],
                                  verbose_eval=False)
                y_pred = model.predict(x_val_cv)

                # Classification task
                if self.ml_algorithm == 'LGBMClassifier':

                    # Binary classification: round to the nearest integer
                    if num_classes <= 2:

                        y_pred = np.rint(y_pred)

                    # Multiclass classification: identify the predicted class based on the one-hot-encoded probabilities
                    else:

                        y_one_hot_proba = np.copy(y_pred)
                        n_rows = y_one_hot_proba.shape[0]

                        y_pred = np.zeros(shape=(n_rows, 1))

                        # Identify the predicted class for each row (highest probability)
                        for row in range(n_rows):
                            y_pred[row, 0] = np.argmax(y_one_hot_proba[row, :])

                # Add remaining ML-algorithms here

            else:
                raise Exception('Unknown ML-algorithm!')

            # Compute the warmstart (validation) loss according to the loss_metric selected
            warmstart_loss = self.metric(y_val_cv, y_pred)

            cross_val_losses.append(warmstart_loss)

        # Compute the average cross validation loss for the warmstart configuration
        warmstart_loss_cv = np.mean(cross_val_losses)

        return warmstart_loss_cv

    def train_evaluate_ml_model(self, params, cv_mode=True, test_mode=False, **kwargs):
        """
        Method serves as superior logic layer for the different train_evalute_<ML-library>_model(...) methods.
        The method selects the selected ML algorithm and initiates the training based on the hyperparameter
        configuration that is specified by params. Returns the loss.
        :param params: dict
            Dictionary of hyperparameters
        :param cv_mode: bool
            Flag that indicates, whether to perform cross validation or simple validation
        :param test_mode: bool
            Flag that indicates, whether to compute the loss on the test set or not
        :param kwargs: dict
            Further keyword arguments (e.g. hp_budget: share of training set (x_train, y_train))
        :return: loss
            Loss of this evaluation
        """

        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR' or \
                self.ml_algorithm == 'AdaBoostRegressor' or self.ml_algorithm == 'DecisionTreeRegressor' or \
                self.ml_algorithm == 'LinearRegression' or self.ml_algorithm == 'KNNRegressor' or \
                self.ml_algorithm == 'RandomForestClassifier' or self.ml_algorithm == 'SVC' or \
                self.ml_algorithm == 'LogisticRegression' or self.ml_algorithm == 'NaiveBayes' or \
                self.ml_algorithm == 'DecisionTreeClassifier' or self.ml_algorithm == 'KNNClassifier' or \
                self.ml_algorithm == 'AdaBoostClassifier' or self.ml_algorithm == 'MLPClassifier' or \
                self.ml_algorithm == 'MLPRegressor' or self.ml_algorithm == 'ElasticNet':

            eval_func = self.train_evaluate_scikit_model

        elif self.ml_algorithm == 'KerasRegressor' or self.ml_algorithm == 'KerasClassifier':

            eval_func = self.train_evaluate_keras_model

        elif self.ml_algorithm == 'XGBoostRegressor' or self.ml_algorithm == 'XGBoostClassifier':

            eval_func = self.train_evaluate_xgboost_model

        elif self.ml_algorithm == 'LGBMRegressor' or self.ml_algorithm == 'LGBMClassifier':
            eval_func = self.train_evaluate_lightgbm_model

        else:
            raise Exception('Unknown ML-algorithm!')

        loss = eval_func(params=params, cv_mode=cv_mode, test_mode=test_mode, **kwargs)

        return loss

    def train_evaluate_scikit_model(self, params: dict, cv_mode=True, test_mode=False, **kwargs):
        """
        This method trains a scikit-learn model according to the selected HP-configuration and returns the
        validation loss
        :param cv_mode: bool
            Flag that indicates, whether to perform cross validation or simple validation
        :param test_mode: bool
            Flag that indicates, whether to compute the loss on the test set or not
        :param params: dict
            Dictionary of hyperparameters
        :param kwargs: dict
            Further keyword arguments (e.g. hp_budget: share of training set (x_train, y_train))
        :return: cv_loss: float
            Validation loss of this run
        """

        # Preprocess "bidimensional" MLP parameter
        if self.ml_algorithm == 'MLPRegressor' or self.ml_algorithm == 'MLPClassifier':
            n_hidden_layers = params.pop('n_hidden_layers')
            hidden_layer_size = params.pop('hidden_layer_size')

            params['hidden_layer_sizes'] = (hidden_layer_size,) * n_hidden_layers

        # Create K-Folds cross validator
        kf = KFold(n_splits=5)
        cross_val_losses = []
        cv_iter = 0

        # Iterate over the cross validation splits
        for train_index, val_index in kf.split(X=self.x_train):
            cv_iter = cv_iter + 1

            # Cross validation
            if cv_mode and not test_mode:

                x_train_cv, x_val_cv = self.x_train.iloc[train_index], self.x_train.iloc[val_index]
                y_train_cv, y_val_cv = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            # Separate a validation set, but do not perform cross validation
            elif not cv_mode and not test_mode and cv_iter < 2:

                x_train_cv, x_val_cv, y_train_cv, y_val_cv = train_test_split(self.x_train, self.y_train, test_size=0.2,
                                                                              shuffle=True, random_state=0)

            # Training on full training set and evaluation on test set
            elif not cv_mode and test_mode and cv_iter < 2:

                x_train_cv, x_val_cv = self.x_train, self.x_test
                y_train_cv, y_val_cv = self.y_train, self.y_test

            elif cv_mode and test_mode:

                raise Exception('Cross validation is not implemented for test mode.')

            # Iteration doesn't make sense for non cross validation
            else:
                continue

            # Create ML-model for the HP-configuration selected by the HPO-method
            if self.ml_algorithm == 'MLPRegressor':
                model = MLPRegressor(**params, random_state=self.random_seed, verbose=True)

            elif self.ml_algorithm == 'MLPClassifier':
                model = MLPClassifier(**params, random_state=self.random_seed, verbose=True)

            elif self.ml_algorithm == 'RandomForestRegressor':
                model = RandomForestRegressor(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'RandomForestClassifier':
                model = RandomForestClassifier(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'SVR':
                # SVR has no random_state parameter
                # Increase cache size to 500 MB to avoid slow trainings due to cache shortages
                params['cache_size'] = 500
                model = SVR(**params)

            elif self.ml_algorithm == 'SVC':
                # Increase cache size to 500 MB to avoid slow trainings due to cache shortages
                params['cache_size'] = 500
                model = SVC(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'AdaBoostRegressor':
                # Set the max_depth of the base estimator object
                max_depth = params.pop('max_depth')
                params['base_estimator'] = DecisionTreeRegressor(max_depth=max_depth)

                model = AdaBoostRegressor(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'AdaBoostClassifier':
                # Set the max_depth of the base estimator object
                max_depth = params.pop('max_depth')
                params['base_estimator'] = DecisionTreeClassifier(max_depth=max_depth)

                model = AdaBoostClassifier(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'DecisionTreeRegressor':
                model = DecisionTreeRegressor(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'DecisionTreeClassifier':
                model = DecisionTreeClassifier(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'LinearRegression':
                # LinearRegression has no random_state parameter
                model = LinearRegression(**params)

            elif self.ml_algorithm == 'KNNRegressor':
                # KNeighborsRegressor has no random_state parameter
                model = KNeighborsRegressor(**params)

            elif self.ml_algorithm == 'KNNClassifier':
                # KNeighborsRegressor has no random_state parameter
                model = KNeighborsClassifier(**params)

            elif self.ml_algorithm == 'LogisticRegression':
                model = LogisticRegression(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'NaiveBayes':
                # GaussianNB has no random_state parameter
                model = GaussianNB(**params)

            elif self.ml_algorithm == 'ElasticNet':
                model = ElasticNet(**params, random_state=self.random_seed)

            else:
                raise Exception('Unknown ML-algorithm!')

            if 'hb_budget' in kwargs and not test_mode:
                # For BOHB and Hyperband select the training data according to the budget of this iteration
                hb_budget = kwargs['hb_budget']
                n_train = len(x_train_cv)
                n_budget = int(0.1 * hb_budget * n_train)
                idx_train = np.random.randint(low=0, high=n_budget, size=n_budget)
                x_train_cv = x_train_cv.iloc[idx_train]
                y_train_cv = y_train_cv.iloc[idx_train]

            elif 'fabolas_budget' in kwargs and not test_mode:
                # For Fabolas select the training data according to the budget of this iteration
                fabolas_budget = kwargs['fabolas_budget']
                idx_train = np.random.randint(low=0, high=fabolas_budget, size=fabolas_budget)
                x_train_cv = x_train_cv.iloc[idx_train]
                y_train_cv = y_train_cv.iloc[idx_train]

            # Train the model and make the prediction
            model.fit(x_train_cv, y_train_cv)
            y_pred = model.predict(x_val_cv)

            # Compute the validation loss according to the loss_metric selected
            val_loss = self.metric(y_val_cv, y_pred)

            cross_val_losses.append(val_loss)

        if not test_mode:

            # Measure the finish time of the iteration
            self.times.append(time.time())

            # Compute the average cross validation loss
            cv_loss = np.mean(cross_val_losses)

        else:
            cv_loss = cross_val_losses[0]

        return cv_loss

    def train_evaluate_keras_model(self, params: dict, cv_mode=True, test_mode=False, **kwargs):
        """
        This method trains a keras model according to the selected HP-configuration and returns the
        validation loss
        :param params: dict
            Dictionary of hyperparameters
        :param cv_mode: bool
            Flag that indicates, whether to perform cross validation or simple validation
        :param test_mode: bool
            Flag that indicates, whether to compute the loss on the test set or not
        :param kwargs: dict
            Further keyword arguments (e.g. hp_budget: share of the total number of epochs for training)
        :return: val_loss: float
            Validation loss of this run
        """
        full_budget_epochs = 100  # see https://arxiv.org/abs/1905.04970

        # Create K-Folds cross validator
        kf = KFold(n_splits=5)
        cross_val_losses = []
        cv_iter = 0

        # Iterate over the cross validation splits
        for train_index, val_index in kf.split(X=self.x_train):
            cv_iter = cv_iter + 1

            # Cross validation
            if cv_mode and not test_mode:

                x_train_cv, x_val_cv = self.x_train.iloc[train_index], self.x_train.iloc[val_index]
                y_train_cv, y_val_cv = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            # Separate a validation set, but do not perform cross validation
            elif not cv_mode and not test_mode and cv_iter < 2:

                x_train_cv, x_val_cv, y_train_cv, y_val_cv = train_test_split(self.x_train, self.y_train, test_size=0.2,
                                                                              shuffle=True, random_state=0)

            # Training on full training set and evaluation on test set
            elif not cv_mode and test_mode and cv_iter < 2:

                x_train_cv, x_val_cv = self.x_train, self.x_test
                y_train_cv, y_val_cv = self.y_train, self.y_test

            elif cv_mode and test_mode:

                raise Exception('Cross validation is not implemented for test mode.')

            # Iteration doesn't make sense for non cross validation
            else:
                continue

            if 'hb_budget' in kwargs and not test_mode:
                # For BOHB and Hyperband select the training data according to the budget of this iteration
                hb_budget = kwargs['hb_budget']
                n_train = len(x_train_cv)
                n_budget = int(0.1 * hb_budget * n_train)
                idx_train = np.random.randint(low=0, high=n_budget, size=n_budget)
                x_train_cv = x_train_cv.iloc[idx_train]
                y_train_cv = y_train_cv.iloc[idx_train]
                epochs = full_budget_epochs

            elif 'fabolas_budget' in kwargs and not test_mode:
                # For Fabolas select the training data according to the budget of this iteration
                fabolas_budget = kwargs['fabolas_budget']
                idx_train = np.random.randint(low=0, high=fabolas_budget, size=fabolas_budget)
                x_train_cv = x_train_cv.iloc[idx_train]
                y_train_cv = y_train_cv.iloc[idx_train]
                epochs = full_budget_epochs

            else:
                epochs = full_budget_epochs  # train on the full budget

            # Set the (global) random seed for TensorFlow
            set_seed(self.random_seed)

            # Initialize the neural network
            model = keras.Sequential()

            # Add input layer
            model.add(keras.layers.InputLayer(input_shape=len(x_train_cv.keys())))

            # Add first hidden layer
            if params['hidden_layer1_size'] > 0:
                model.add(
                    keras.layers.Dense(params['hidden_layer1_size'], activation=params['hidden_layer1_activation']))
                model.add(keras.layers.Dropout(params['dropout1']))

            # Add second hidden layer
            if params['hidden_layer2_size'] > 0:
                model.add(
                    keras.layers.Dense(params['hidden_layer2_size'], activation=params['hidden_layer2_activation']))
                model.add(keras.layers.Dropout(params['dropout2']))

            # Add third hidden layer
            if params['hidden_layer3_size'] > 0:
                model.add(
                    keras.layers.Dense(params['hidden_layer3_size'], activation=params['hidden_layer3_activation']))
                model.add(keras.layers.Dropout(params['dropout3']))

            # Add fourth hidden layer
            if params['hidden_layer4_size'] > 0:
                model.add(
                    keras.layers.Dense(params['hidden_layer4_size'], activation=params['hidden_layer4_activation']))
                model.add(keras.layers.Dropout(params['dropout4']))

            # Add output layer
            if self.ml_algorithm == 'KerasRegressor':

                model.add(keras.layers.Dense(1, activation='linear'))

                # Select optimizer and compile the model
                adam = keras.optimizers.Adam(learning_rate=params['init_lr'])
                model.compile(optimizer=adam, loss='mse', metrics=['mse'])

            elif self.ml_algorithm == 'KerasClassifier':

                # Determine the number of different classes depending on the data format
                if type(y_train_cv) == pd.core.series.Series:
                    num_classes = int(max(y_train_cv) - min(y_train_cv) + 1)

                elif type(y_train_cv) == pd.core.frame.DataFrame:
                    num_classes = len(y_train_cv.keys())

                else:
                    raise Exception('Unknown data format!')

                # Binary classification
                if num_classes <= 2:

                    # 'Sigmoid is equivalent to a 2-element Softmax, where the second element is assumed to be zero'
                    # https://keras.io/api/layers/activations/#sigmoid-function
                    model.add(keras.layers.Dense(1, activation='sigmoid'))

                    adam = keras.optimizers.Adam(learning_rate=params['init_lr'])
                    model.compile(optimizer=adam, loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

                    # TODO: Add class weights for imbalanced classification problems
                    # # Calculate class weights
                    # n_zero = sum(y_train_cv == 0)
                    # n_one = sum(y_train_cv == 1)
                    #
                    # weight_zero = (1/n_zero) * (n_zero + n_one)
                    # weight_one = (1/n_one) * (n_zero + n_one)
                    # class_weight = {0: weight_zero, 1: weight_one}
                    #
                    # weight_dict = {'class_weight': class_weight}

                # Multiclass classification
                else:

                    # Use softmax activation for multiclass clf. -> 'Softmax converts a real vector to a vector of
                    # categorical probabilities. [...] the result could be interpreted as a probability distribution.'
                    # https://keras.io/api/layers/activations/#softmax-function
                    model.add(keras.layers.Dense(num_classes, activation='softmax'))

                    adam = keras.optimizers.Adam(learning_rate=params['init_lr'])
                    model.compile(optimizer=adam, loss=keras.losses.CategoricalCrossentropy(),
                                  metrics=[keras.metrics.CategoricalAccuracy()])

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

            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           min_delta=0,
                                                           patience=10,
                                                           verbose=0,
                                                           mode='auto',
                                                           restore_best_weights=True)

            callbacks_list = [lr, early_stopping]

            # Train the model and make the prediction
            model.fit(x_train_cv, y_train_cv, epochs=epochs, batch_size=params['batch_size'],
                      validation_data=(x_val_cv, y_val_cv), callbacks=callbacks_list,
                      verbose=0)

            # Make the prediction
            y_pred = model.predict(x_val_cv)

            # In case of binary classification round to the nearest integer
            if self.ml_algorithm == 'KerasClassifier':

                # Binary classification
                if num_classes <= 2:

                    y_pred = np.rint(y_pred)

                # Multiclass classification
                else:

                    # Identify the predicted class (maximum probability) in each row
                    for row_idx in range(y_pred.shape[0]):

                        # Predicted class
                        this_class = np.argmax(y_pred[row_idx, :])

                        # Iterate over columns / classes
                        for col_idx in range(y_pred.shape[1]):

                            if col_idx == this_class:
                                y_pred[row_idx, col_idx] = 1
                            else:
                                y_pred[row_idx, col_idx] = 0

            # KerasRegressor
            else:
                y_pred = np.reshape(y_pred, newshape=(-1,))

            # Compute the validation loss according to the loss_metric selected
            val_loss = self.metric(y_val_cv, y_pred)

            cross_val_losses.append(val_loss)

        if not test_mode:

            # Measure the finish time of the iteration
            self.times.append(time.time())

            # Compute the average cross validation loss
            cv_loss = np.mean(cross_val_losses)

        else:
            cv_loss = cross_val_losses[0]

        return cv_loss

    def train_evaluate_xgboost_model(self, params: dict, cv_mode=True, test_mode=False, **kwargs):
        """
        This method trains a XGBoost model according to the selected HP-configuration and returns the
        validation loss
        :param cv_mode: bool
            Flag that indicates, whether to perform cross validation or simple validation
        :param test_mode: bool
            Flag that indicates, whether to compute the loss on the test set or not
        :param params: dict
            Dictionary of hyperparameters
        :param kwargs: dict
            Further keyword arguments (e.g. hp_budget: share of training set (x_train, y_train))
        :return: val_loss: float
            Validation loss of this run
        """

        # Consideration of conditional hyperparameters // Remove invalid HPs according to the conditions
        if params['booster'] not in ['gbtree', 'dart']:
            del params['eta']
            del params['subsample']
            del params['max_depth']
            del params['min_child_weight']
            del params['colsample_bytree']
            del params['colsample_bylevel']

        # Create K-Folds cross validator
        kf = KFold(n_splits=5)
        cross_val_losses = []
        cv_iter = 0

        # Iterate over the cross validation splits
        for train_index, val_index in kf.split(X=self.x_train):
            cv_iter = cv_iter + 1

            # Cross validation
            if cv_mode and not test_mode:

                x_train_cv, x_val_cv = self.x_train.iloc[train_index], self.x_train.iloc[val_index]
                y_train_cv, y_val_cv = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            # Separate a validation set, but do not perform cross validation
            elif not cv_mode and not test_mode and cv_iter < 2:

                x_train_cv, x_val_cv, y_train_cv, y_val_cv = train_test_split(self.x_train, self.y_train, test_size=0.2,
                                                                              shuffle=True, random_state=0)

            # Training on full training set and evaluation on test set
            elif not cv_mode and test_mode and cv_iter < 2:

                x_train_cv, x_val_cv = self.x_train, self.x_test
                y_train_cv, y_val_cv = self.y_train, self.y_test

            elif cv_mode and test_mode:

                raise Exception('Cross validation is not implemented for test mode.')

            # Iteration doesn't make sense for non cross validation
            else:
                continue

            if 'hb_budget' in kwargs and not test_mode:
                # For BOHB and Hyperband select the training data according to the budget of this iteration
                hb_budget = kwargs['hb_budget']
                n_train = len(x_train_cv)
                n_budget = int(0.1 * hb_budget * n_train)
                idx_train = np.random.randint(low=0, high=n_budget, size=n_budget)
                x_train_cv = x_train_cv.iloc[idx_train]
                y_train_cv = y_train_cv.iloc[idx_train]

            elif 'fabolas_budget' in kwargs and not test_mode:
                # For Fabolas select the training data according to the budget of this iteration
                fabolas_budget = kwargs['fabolas_budget']
                idx_train = np.random.randint(low=0, high=fabolas_budget, size=fabolas_budget)
                x_train_cv = x_train_cv.iloc[idx_train]
                y_train_cv = y_train_cv.iloc[idx_train]

            # Initialize the model
            if self.ml_algorithm == 'XGBoostRegressor':

                model = XGBRegressor(**params, random_state=self.random_seed)

            elif self.ml_algorithm == 'XGBoostClassifier':

                model = XGBClassifier(**params, random_state=self.random_seed)

            # Train the model and make the prediction
            model.fit(x_train_cv, y_train_cv)
            y_pred = model.predict(x_val_cv)

            # Compute the validation loss according to the loss_metric selected
            val_loss = self.metric(y_val_cv, y_pred)

            cross_val_losses.append(val_loss)

        if not test_mode:

            # Measure the finish time of the iteration
            self.times.append(time.time())

            # Compute the average cross validation loss
            cv_loss = np.mean(cross_val_losses)

        else:
            cv_loss = cross_val_losses[0]

        return cv_loss

    def train_evaluate_lightgbm_model(self, params, cv_mode=True, test_mode=False, **kwargs):
        """
        This method trains a LightGBM model according to the selected HP-configuration and returns the
        validation loss.
        :param cv_mode: bool
            Flag that indicates, whether to perform cross validation or simple validation
        :param test_mode: bool
            Flag that indicates, whether to compute the loss on the test set or not
        :param params: dict
            Dictionary of hyperparameters
        :param kwargs: dict
            Further keyword arguments (e.g. hp_budget: share of training set (x_train, y_train))
        :return: val_loss: float
            Validation loss of this run
        """

        # Create K-Folds cross validator
        kf = KFold(n_splits=5)
        cross_val_losses = []
        cv_iter = 0

        # Iterate over the cross validation splits
        for train_index, val_index in kf.split(X=self.x_train):
            cv_iter = cv_iter + 1

            # Cross validation
            if cv_mode and not test_mode:

                x_train_cv, x_val_cv = self.x_train.iloc[train_index], self.x_train.iloc[val_index]
                y_train_cv, y_val_cv = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            # Separate a validation set, but do not perform cross validation
            elif not cv_mode and not test_mode and cv_iter < 2:

                x_train_cv, x_val_cv, y_train_cv, y_val_cv = train_test_split(self.x_train, self.y_train, test_size=0.2,
                                                                              shuffle=True, random_state=0)

            # Training on full training set and evaluation on test set
            elif not cv_mode and test_mode and cv_iter < 2:

                x_train_cv, x_val_cv = self.x_train, self.x_test
                y_train_cv, y_val_cv = self.y_train, self.y_test

            elif cv_mode and test_mode:

                raise Exception('Cross validation is not implemented for test mode.')

            # Iteration doesn't make sense for non cross validation
            else:
                continue

            if 'hb_budget' in kwargs and not test_mode:
                # For BOHB and Hyperband select the training data according to the budget of this iteration
                hb_budget = kwargs['hb_budget']
                n_train = len(x_train_cv)
                n_budget = int(0.1 * hb_budget * n_train)
                idx_train = np.random.randint(low=0, high=n_budget, size=n_budget)
                x_train_cv = x_train_cv.iloc[idx_train]
                y_train_cv = y_train_cv.iloc[idx_train]

            elif 'fabolas_budget' in kwargs and not test_mode:
                # For Fabolas select the training data according to the budget of this iteration
                fabolas_budget = kwargs['fabolas_budget']
                idx_train = np.random.randint(low=0, high=fabolas_budget, size=fabolas_budget)
                x_train_cv = x_train_cv.iloc[idx_train]
                y_train_cv = y_train_cv.iloc[idx_train]

            # Specify the Ml task
            if self.ml_algorithm == 'LGBMRegressor':
                # Regression task
                params['objective'] = 'regression'

            elif self.ml_algorithm == 'LGBMClassifier':

                # Determine the number of classes
                num_classes = int(max(y_train_cv) - min(y_train_cv) + 1)

                # Binary classification task
                if num_classes <= 2:
                    params['objective'] = 'binary'

                # Multiclass classification task
                else:
                    params['objective'] = 'multiclass'  # uses Softmax objective function
                    params['num_class'] = num_classes

            # Specify random seed
            params['seed'] = self.random_seed
            params['verbosity'] = -1

            # Create lgb datasets
            train_data = lgb.Dataset(x_train_cv, label=y_train_cv)
            valid_data = lgb.Dataset(x_val_cv, label=y_val_cv)

            # Initialize and train the model
            lgb_model = lgb.train(params=params, train_set=train_data, valid_sets=[valid_data], verbose_eval=False)

            # Make the prediction
            y_pred = lgb_model.predict(data=x_val_cv)

            # Classification task
            if self.ml_algorithm == 'LGBMClassifier':

                # Binary classification: round to the nearest integer
                if num_classes <= 2:

                    y_pred = np.rint(y_pred)

                # Multiclass classification: identify the predicted class based on the one-hot-encoded probabilities
                else:

                    y_one_hot_proba = np.copy(y_pred)
                    n_rows = y_one_hot_proba.shape[0]

                    y_pred = np.zeros(shape=(n_rows, 1))

                    # Identify the predicted class for each row (highest probability)
                    for row in range(n_rows):
                        y_pred[row, 0] = np.argmax(y_one_hot_proba[row, :])

            # Compute the validation loss according to the loss_metric selected
            val_loss = self.metric(y_val_cv, y_pred)

            cross_val_losses.append(val_loss)

        if not test_mode:

            # Measure the finish time of the iteration
            self.times.append(time.time())

            # Compute the average cross validation loss
            cv_loss = np.mean(cross_val_losses)

        else:
            cv_loss = cross_val_losses[0]

        return cv_loss
