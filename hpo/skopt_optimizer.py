import skopt
from skopt.optimizer import forest_minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow import keras
import matplotlib.pyplot as plt
import functools
import time

from hpo.lr_schedules import fix, exponential, cosine
from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class SkoptOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget,
                         random_seed)

    def train_evaluate_scikit_regressor(self, params):
        """ This method trains a scikit-learn model according to the selected HP-configuration and returns the
        validation loss"""

        # Create ML-model for the HP-configuration selected by the HPO-method
        if self.ml_algorithm == 'RandomForestRegressor':
            model = RandomForestRegressor(**params, random_state=self.random_seed)
        elif self.ml_algorithm == 'SVR':
            model = SVR(**params)  # SVR has no random_state argument

        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_val)

        # Compute the validation loss according to the metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss

    def objective_scikit_regressor(self, params):
        # Objective function for a scikit-learn regressor

        # Convert the hyperparameters into a dictionary to pass them to the ML-model
        dict_params = {}
        for i in range(len(self.hp_space)):
            dict_params[self.hp_space[i].name] = params[i]

        return self.train_evaluate_scikit_regressor(dict_params)

    def train_evaluate_keras_regressor(self, params):
        """ This method trains a Keras model according to the selected HP-configuration and returns the
        validation loss"""

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

        if params["lr_schedule"] == "exponential":
            schedule = functools.partial(exponential, initial_lr=params["init_lr"], T_max=self.budget)

        elif params["lr_schedule"] == "constant":
            schedule = functools.partial(fix, initial_lr=params["init_lr"])

        lr = keras.callbacks.LearningRateScheduler(schedule)

        callbacks_list = [lr]

        model.fit(self.x_train, self.y_train, epochs=self.budget, batch_size=params['batch_size'],
                  validation_data=(self.x_val, self.y_val), callbacks=callbacks_list,
                  verbose=1)

        y_pred = model.predict(self.x_val)

        # Compute the validation loss according to the metric selected
        val_loss = self.metric(self.y_val, y_pred)

        # Measure the finish time of the iteration
        self.times.append(time.time())

        return val_loss

    def objective_keras_regressor(self, params):
        # Objective function for a Keras regressor

        # Convert the hyperparameters into a dictionary to pass them to the ML-model
        dict_params = {}
        for i in range(len(self.hp_space)):
            dict_params[self.hp_space[i].name] = params[i]

        return self.train_evaluate_keras_regressor(dict_params)

    def optimize(self) -> TuningResult:

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR':
            this_objective = self.objective_scikit_regressor
        elif self.ml_algorithm == 'KerasRegressor':
            this_objective = self.objective_keras_regressor
        else:
            raise NameError('Unknown ML-algorithm!')

        # Select the specified HPO-tuning method
        if self.hpo_method == 'SMAC':
            this_optimizer = forest_minimize
            this_acq_func = 'EI'
        else:
            raise NameError('Unknown HPO-method!')

        # Optimize on the predefined budget and measure the optimization time
        start_time = time.time()
        self.times = []  # Initialize a list for saving the optimization times

        # Start the optimization
        trial_result = this_optimizer(this_objective, self.hp_space, n_calls=self.budget, random_state=self.random_seed,
                                      acq_func=this_acq_func)

        # wall_clock_time = time.time() - start_time
        for i in range(len(self.times)):
            # Subtract to start time to receive the durations of each evaluation
            self.times[i] = self.times[i] - start_time
        wall_clock_time = max(self.times)

        # Create a TuningResult-object to store the optimization results
        # Transformation of the results into a TuningResult-Object
        best_loss = trial_result.fun
        losses = list(trial_result.func_vals)

        best_configuration = {}
        for i in range(len(self.hp_space)):
            best_configuration[self.hp_space[i].name] = trial_result.x[i]

        evaluation_ids = list(range(1, len(trial_result.func_vals) + 1))
        configurations = ()
        for i in range(len(trial_result.x_iters)):
            this_config = {}
            for j in range(len(self.hp_space)):
                this_config[self.hp_space[j].name] = trial_result.x_iters[i][j]
            configurations = configurations + (this_config,)

        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=self.times, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_configuration,
                              wall_clock_time=wall_clock_time)

        return result
