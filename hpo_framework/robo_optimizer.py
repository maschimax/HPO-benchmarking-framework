import numpy as np
import skopt
import time
from robo.fmin import bayesian_optimization, fabolas

from hpo_framework.baseoptimizer import BaseOptimizer
from hpo_framework.results import TuningResult


class RoboOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                 random_seed, n_workers, do_warmstart):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                         random_seed, n_workers)

        self.do_warmstart = do_warmstart

    def optimize(self) -> TuningResult:
        """
        Method performs a hyperparameter optimization run according to the selected HPO-method.
        :return: result: TuningResult
            TuningResult-object that contains the results of this optimization run.
        :return:
        """

        # Convert the skopt hyperparameter space into a continuous space for RoBO
        hp_space_lower = np.zeros(shape=(len(self.hp_space),))
        hp_space_upper = np.zeros(shape=(len(self.hp_space),))

        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                hp_space_lower[i, ] = self.hp_space[i].low
                hp_space_upper[i, ] = self.hp_space[i].high

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                n_choices = len(list(self.hp_space[i].categories))
                hp_space_lower[i, ] = 0
                hp_space_upper[i, ] = n_choices - 1

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                hp_space_lower[i, ] = self.hp_space[i].low
                hp_space_upper[i, ] = self.hp_space[i].high

            else:
                raise Exception('The skopt HP-space could not be converted correctly!')

        # Set the random seed of the random number generator
        rand_num_generator = np.random.RandomState(seed=self.random_seed)

        # Optimize on the predefined n_func_evals and measure the wall clock times
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Use a warmstart configuration (only possible for BOHAMIANN, not FABOLAS)
        if self.do_warmstart == 'Yes' and self.hpo_method != 'Fabolas':

            # Initialize numpy arrays for saving the warmstart configuration and the warmstart loss
            warmstart_config = np.zeros(shape=(1, len(self.hp_space)))
            warmstart_loss = np.zeros(shape=(1, 1))

            # Retrieve the default hyperparameters and the default loss for the ML-algorithm
            default_params = self.get_warmstart_configuration()

            try:

                # Dictionary for saving the warmstart HP-configuration (only contains the HPs, which are part of the
                # 'tuned' HP-space
                warmstart_dict = {}

                # Iterate over all HPs of this ML-algorithm's tuned HP-space and append the default values to
                # the numpy array
                for i in range(len(self.hp_space)):

                    this_param = self.hp_space[i].name

                    # Categorical HPs need to be encoded as integer values for RoBO
                    if type(self.hp_space[i]) == skopt.space.space.Categorical:

                        choices = self.hp_space[i].categories
                        this_warmstart_value_cat = default_params[this_param]
                        dict_value = this_warmstart_value_cat

                        # Find the index of the default / warmstart HP in the list of possible choices
                        for j in range(len(choices)):
                            if this_warmstart_value_cat == choices[j]:
                                this_warmstart_value = j

                    # For all non-categorical HPs
                    else:
                        this_warmstart_value = default_params[this_param]
                        dict_value = this_warmstart_value

                        # For some HPs (e.g. max_depth of RF) the default value is None, although their typical dtype is
                        # different (e.g. int)
                        if this_warmstart_value is None and type(self.hp_space[i]) == skopt.space.space.Integer:
                            # Try to impute these values by the mean value
                            this_warmstart_value = int(0.5 * (self.hp_space[i].low + self.hp_space[i].high))
                            dict_value = this_warmstart_value

                    # Pass the warmstart value to the according numpy array
                    warmstart_config[0, i] = this_warmstart_value
                    warmstart_dict[this_param] = dict_value

                # Pass the default loss to the according numpy array
                warmstart_loss[0, 0] = self.get_warmstart_loss(warmstart_dict=warmstart_dict)

                # Pass the warmstart configuration as a kwargs dict
                kwargs = {'X_init': warmstart_config,
                          'Y_init': warmstart_loss}

                # Set flag to indicate that a warmstart took place
                did_warmstart = True

            except:
                print('Warmstarting RoBO failed!')
                kwargs = {}

                # Set flag to indicate that NO warmstart took place
                did_warmstart = False

        # No warmstart requested
        else:
            kwargs = {}

            # Set flag to indicate that NO warmstart took place
            did_warmstart = False

        # Select the specified HPO-tuning method
        try:
            if self.hpo_method == 'Fabolas':

                # Budget correct? // Set further parameters?
                s_max = len(self.x_train)  # Maximum number of data points for the training data set
                s_min = int(0.05 * s_max)  # Maximum number of data points for the training data set
                n_init = int(self.n_func_evals / 3)  # Requirement of the fabolas implementation

                result_dict = fabolas(objective_function=self.objective_fabolas, s_min=s_min, s_max=s_max,
                                      lower=hp_space_lower, upper=hp_space_upper,
                                      num_iterations=self.n_func_evals, rng=rand_num_generator, n_init=n_init)
                run_successful = True

            elif self.hpo_method == 'Bohamiann':

                if did_warmstart:
                    # A single initial design point (warm start hyperparameter configuration)
                    kwargs['n_init'] = 1

                # Budget correct? // Set further parameters?
                result_dict = bayesian_optimization(objective_function=self.objective_bohamiann,
                                                    lower=hp_space_lower, upper=hp_space_upper,
                                                    model_type='bohamiann', num_iterations=self.n_func_evals,
                                                    rng=rand_num_generator, **kwargs)
                run_successful = True

            else:
                raise Exception('Unknown HPO-method!')

        # Algorithm crashed
        except:
            # Add a warning here
            run_successful = False

        # If the optimization run was successful, determine the optimization results
        if run_successful:

            for i in range(len(self.times)):
                # Subtract the start time to receive the wall clock time of each function evaluation
                self.times[i] = self.times[i] - start_time
            wall_clock_time = max(self.times)

            # Insert timestamp of 0.0 for the warm start hyperparameter configuration
            if did_warmstart:
                self.times.insert(0, 0.0)

            # Timestamps
            timestamps = self.times

            # Losses (not incumbent losses)
            losses = result_dict['y']

            evaluation_ids = list(range(1, len(losses) + 1))
            best_loss = min(losses)

            configurations = ()
            for config in result_dict['X']:
                # Cut off the unused Fabolas budget value at the end
                config = config[:len(self.hp_space)]
                config_dict = {}

                for i in range(len(config)):
                    if type(self.hp_space[i]) == skopt.space.space.Integer:
                        config_dict[self.hp_space[i].name] = int(round(config[i]))

                    elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                        config_dict[self.hp_space[i].name] = list(self.hp_space[i].categories)[int(round(config[i]))]

                    elif type(self.hp_space[i]) == skopt.space.space.Real:
                        config_dict[self.hp_space[i].name] = config[i]

                    else:
                        raise Exception('The continuous HP-space could not be converted correctly!')

                configurations = configurations + (config_dict,)

            # Find the best hyperparameter configuration (incumbent)
            best_configuration = {}
            x_opt = result_dict['x_opt']

            for i in range(len(x_opt)):
                if type(self.hp_space[i]) == skopt.space.space.Integer:
                    best_configuration[self.hp_space[i].name] = int(round(x_opt[i]))

                elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                    best_configuration[self.hp_space[i].name] = list(self.hp_space[i].categories)[int(round(x_opt[i]))]

                elif type(self.hp_space[i]) == skopt.space.space.Real:
                    best_configuration[self.hp_space[i].name] = x_opt[i]

                else:
                    raise Exception('The continuous HP-space could not be converted correctly!')

        # Run not successful (algorithm crashed)
        else:
            evaluation_ids, timestamps, losses, configurations, best_loss, best_configuration, wall_clock_time = \
                self.impute_results_for_crash()

        # Pass the results to a TuningResult-Object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=timestamps, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_configuration,
                              wall_clock_time=wall_clock_time, successful=run_successful, did_warmstart=did_warmstart)

        return result

    def objective_fabolas(self, cont_hp_space, s):
        """
        Objective function for FABOLAS: This method converts the given hyperparameters into a dictionary, passes them
        to the ML-model for training and returns the validation loss and the evaluation time (cost).
        :param cont_hp_space: np.array
            Array that contains the next hyperparameter configuration (continuous) to be evaluated
        :param s:
            Fraction of s_max
        :return: val_loss: float, cost: float
            Validation loss and evaluation cost (time)
        """

        # Transform the !continuous! hyperparameters into their respective types and save them in a dictionary
        dict_params = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                dict_params[self.hp_space[i].name] = int(round(cont_hp_space[i, ]))

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                dict_params[self.hp_space[i].name] = list(self.hp_space[i].categories)[int(round(cont_hp_space[i, ]))]

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                dict_params[self.hp_space[i].name] = cont_hp_space[i, ]

            else:
                raise Exception('The continuous HP-space could not be converted correctly!')

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR' or \
                self.ml_algorithm == 'AdaBoostRegressor' or self.ml_algorithm == 'DecisionTreeRegressor' or \
                self.ml_algorithm == 'LinearRegression' or self.ml_algorithm == 'KNNRegressor' or \
                self.ml_algorithm == 'RandomForestClassifier' or self.ml_algorithm == 'SVC' or \
                self.ml_algorithm == 'LogisticRegression' or self.ml_algorithm == 'NaiveBayes':

            eval_func = self.train_evaluate_scikit_model

        elif self.ml_algorithm == 'KerasRegressor' or self.ml_algorithm == 'KerasClassifier':
            eval_func = self.train_evaluate_keras_model

        elif self.ml_algorithm == 'XGBoostRegressor' or self.ml_algorithm == 'XGBoostClassifier':
            eval_func = self.train_evaluate_xgboost_model

        elif self.ml_algorithm == 'LGBMRegressor' or self.ml_algorithm == 'LGBMClassifier':
            eval_func = self.train_evaluate_lightgbm_model

        else:
            raise Exception('Unknown ML-algorithm!')

        t_start_eval = time.time()
        val_loss = eval_func(params=dict_params, fabolas_budget=s)
        cost = time.time() - t_start_eval

        return val_loss, cost

    def objective_bohamiann(self, cont_hp_space):
        """
        Objective function for BOHAMIANN: This method converts the given hyperparameters into a dictionary, passes them
        to the ML-model for training and returns the validation loss.
        :param cont_hp_space: np.array
            Array that contains the next hyperparameter configuration (continuous) to be evaluated
        :return: eval_func(params=dict_params): float
            Validation loss for this HP-configuration.
        """

        # Transform the !continuous! hyperparameters into their respective types and save them in a dictionary
        dict_params = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                dict_params[self.hp_space[i].name] = int(round(cont_hp_space[i, ]))

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                dict_params[self.hp_space[i].name] = list(self.hp_space[i].categories)[int(round(cont_hp_space[i, ]))]

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                dict_params[self.hp_space[i].name] = cont_hp_space[i, ]

            else:
                raise Exception('The continuous HP-space could not be converted correctly!')

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR' or \
                self.ml_algorithm == 'AdaBoostRegressor' or self.ml_algorithm == 'DecisionTreeRegressor' or \
                self.ml_algorithm == 'LinearRegression' or self.ml_algorithm == 'KNNRegressor' or \
                self.ml_algorithm == 'RandomForestClassifier' or self.ml_algorithm == 'SVC' or \
                self.ml_algorithm == 'LogisticRegression' or self.ml_algorithm == 'NaiveBayes':

            eval_func = self.train_evaluate_scikit_model

        elif self.ml_algorithm == 'KerasRegressor' or self.ml_algorithm == 'KerasClassifier':
            eval_func = self.train_evaluate_keras_model

        elif self.ml_algorithm == 'XGBoostRegressor' or self.ml_algorithm == 'XGBoostClassifier':
            eval_func = self.train_evaluate_xgboost_model

        elif self.ml_algorithm == 'LGBMRegressor' or self.ml_algorithm == 'LGBMClassifier':
            eval_func = self.train_evaluate_lightgbm_model

        else:
            raise Exception('Unknown ML-algorithm!')

        return eval_func(params=dict_params)
