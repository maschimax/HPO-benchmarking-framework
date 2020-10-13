import optuna
import skopt
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
import pandas as pd
from multiprocessing import Process

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult
from hpo import multiproc_target_funcs


class OptunaOptimizer(BaseOptimizer):
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
        """

        # Select the specified HPO-tuning method
        # if self.hpo_method == 'TPE':
        #     # Create a TPE-Sampler instance with the default parameters of hyperopt
        #     this_optimizer = TPESampler(**TPESampler.hyperopt_parameters(), seed=self.random_seed)

        if self.hpo_method == 'CMA-ES':
            this_optimizer = CmaEsSampler(seed=self.random_seed)

        elif self.hpo_method == 'TPE':
            this_optimizer = TPESampler(seed=self.random_seed)

        elif self.hpo_method == 'RandomSearch':
            this_optimizer = RandomSampler(seed=self.random_seed)

        else:
            raise Exception('Unknown HPO-method!')

        # Create a study object and specify the optimization direction
        study_name = 'hpo_study'
        study_storage = 'sqlite:///hpo.db'

        # Optimize on the predefined n_func_evals and measure the wall clock times
        # start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Delete old study objects ('fresh start') >> otherwise the old results will be included
        try:
            optuna.delete_study(study_name, study_storage)
        except:
            print('No old optuna study objects found!')

        # Use a warmstart configuration?
        if self.do_warmstart == 'Yes':

            try:
                # Create a new study
                warmstart_study = optuna.create_study(direction='minimize', storage=study_storage,
                                                      study_name=study_name,
                                                      load_if_exists=False)

                # Evaluate the warmstart HP-configuration (just a single trial / evaluation)
                warmstart_study.optimize(func=self.warmstart_objective, n_trials=1)

                # Set flag to indicate that a warmstart took place
                did_warmstart = True

            except:
                print('Warmstarting optuna failed!')

                # Set flag to indicate that NO warmstart took place
                did_warmstart = False

        # No warmstart requested
        else:

            # Set flag to indicate that NO warmstart took place
            did_warmstart = False

        # Create a new study or reload the warmstart study (if available and requested)
        study = optuna.create_study(sampler=this_optimizer, direction='minimize',
                                    study_name=study_name, storage=study_storage, load_if_exists=True)

        # Start the optimization
        try:

            # Process based parallelization using multiprocessing
            if self.n_workers > 1:
                # Split the total number of function evaluations between the processes for multiprocessing:
                # First process performs the equal share + the remainder
                n_evals_first_proc = int(self.n_func_evals / self.n_workers) + (self.n_func_evals % self.n_workers)
                # All remaining process perform the equal share of evaluations
                n_evals_remain_proc = int(self.n_func_evals / self.n_workers)

                # Start parallel workers
                processes = []
                for i in range(self.n_workers):

                    if i == 0:
                        n_evals = n_evals_first_proc
                    else:
                        n_evals = n_evals_remain_proc

                    p = Process(target=multiproc_target_funcs.load_study_and_optimize,
                                args=(study_name, study_storage, n_evals, self.objective))

                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

            # No parallelization
            else:
                study.optimize(func=self.objective, n_trials=self.n_func_evals)

            run_successful = True

        # Algorithm crashed
        except:
            # Add a warning here
            run_successful = False

        # If the optimization run was successful, determine the optimization results
        if run_successful:

            # Create a TuningResult-object to store the optimization results
            # Transformation of the results into a TuningResult-Object
            all_trials = study.get_trials()
            best_configuration = study.best_params
            best_loss = study.best_value

            start_times = []  # Start time of each trial
            finish_times = []  # Finish time of each trial
            # evaluation_ids = []  # Number the evaluations / iterations of this run
            unsorted_losses = []  # Loss of each iteration
            unsorted_configurations = ()  # HP-configuration of each iteration

            # Number the evaluations / iterations of this run
            evaluation_ids = list(range(1, len(all_trials) + 1))

            for i in range(len(all_trials)):
                start_times.append(all_trials[i].datetime_start)
                finish_times.append(all_trials[i].datetime_complete)

                # evaluation_ids.append(all_trials[i].number)
                unsorted_losses.append(all_trials[i].value)
                unsorted_configurations = unsorted_configurations + (all_trials[i].params,)

            abs_start_time = min(start_times)  # start time of the first trial
            unsorted_timestamps = []
            for i in range(len(start_times)):
                this_time = finish_times[i] - abs_start_time  # time difference to the start of the first trial
                this_timestamp = this_time.total_seconds()  # conversion into float value
                unsorted_timestamps.append(this_timestamp)

            wall_clock_time = max(unsorted_timestamps)

            ids = list(range(1, len(all_trials) + 1))
            temp_dict = {'ids': ids,
                         'timestamps [finished]': unsorted_timestamps,
                         'losses': unsorted_losses,
                         'configurations': unsorted_configurations,
                         }

            unsorted_df = pd.DataFrame.from_dict(data=temp_dict)
            unsorted_df.set_index('ids', inplace=True)
            sorted_df = unsorted_df.sort_values(by=['timestamps [finished]'], ascending=True, inplace=False)

            timestamps = list(sorted_df['timestamps [finished]'])
            losses = list(sorted_df['losses'])
            configurations = tuple(sorted_df['configurations'])

            # Run not successful (algorithm crashed)
        else:
            evaluation_ids, timestamps, losses, configurations, best_loss, best_configuration, wall_clock_time = \
                self.impute_results_for_crash()

        # Pass the results to a TuningResult-object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=timestamps, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_configuration,
                              wall_clock_time=wall_clock_time, successful=run_successful, did_warmstart=did_warmstart)

        return result

    def objective(self, trial):
        """
        Objective function: This method converts the given hyperparameters into a dictionary, passes them to the
        ML-model for training and returns the validation loss.
        :param trial: optuna.trial._trial.Trial
            Optuna Trial object, that suggests the next HP-configuration according to the selected HPO-method.
        :return: eval_func(params=dict_params)
            Validation loss for the HP-configuration.
        """

        # Convert the hyperparameters into a dictionary to pass them to the ML-model
        dict_params = {}
        for i in range(len(self.hp_space)):
            if type(self.hp_space[i]) == skopt.space.space.Integer:
                dict_params[self.hp_space[i].name] = trial.suggest_int(name=self.hp_space[i].name,
                                                                       low=self.hp_space[i].low,
                                                                       high=self.hp_space[i].high)

            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                dict_params[self.hp_space[i].name] = trial.suggest_categorical(name=self.hp_space[i].name,
                                                                               choices=
                                                                               list(self.hp_space[i].categories))

            elif type(self.hp_space[i]) == skopt.space.space.Real:
                dict_params[self.hp_space[i].name] = trial.suggest_float(name=self.hp_space[i].name,
                                                                         low=self.hp_space[i].low,
                                                                         high=self.hp_space[i].high)
            else:
                raise Exception('The skopt HP-space could not be converted correctly!')

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR' or \
                self.ml_algorithm == 'AdaBoostRegressor' or self.ml_algorithm == 'DecisionTreeRegressor':
            eval_func = self.train_evaluate_scikit_regressor

        elif self.ml_algorithm == 'KerasRegressor':
            eval_func = self.train_evaluate_keras_regressor

        elif self.ml_algorithm == 'XGBoostRegressor':
            eval_func = self.train_evaluate_xgboost_regressor

        else:
            raise Exception('Unknown ML-algorithm!')

        return eval_func(params=dict_params)

    def warmstart_objective(self, trial):
        """
        Objective function for evaluating a given warmstart HP-configuration: This method converts the given
        hyperparameters into a dictionary, passes them to the ML-model for training and returns the warmstart
        validation loss.
        :param trial: optuna.trial._trial.Trial
            Optuna Trial object, that suggests the next HP-configuration according to the selected HPO-method.
        :return:
            Validation loss for the warmstart HP-configuration.
        """

        # Retrieve the default hyperparameters for the ML-algorithms
        default_params = self.get_warmstart_configuration()

        # Convert the hyperparameters into a dictionary to pass them to the ML-model
        warmstart_params = {}
        for i in range(len(self.hp_space)):

            this_param = self.hp_space[i].name

            # For some HPs (e.g. max_depth of RF) the default value is None, although their typical dtype is
            # different (e.g. int)
            if default_params[this_param] is None:
                # Try to impute these values by the mean value
                warmstart_params[this_param] = int(0.5 * (self.hp_space[i].low + self.hp_space[i].high))

            # Integer HPs: Set the upper and lower bound to the default / warmstart value to leave no choice
            elif type(self.hp_space[i]) == skopt.space.space.Integer:
                warmstart_params[this_param] = trial.suggest_int(name=this_param,
                                                                 low=default_params[this_param],
                                                                 high=default_params[this_param])

            # Categorical HPs (warmstarting doesn't work here (see below))
            elif type(self.hp_space[i]) == skopt.space.space.Categorical:
                # Optuna's CategoricalDistribution doesn't support dynamic value spaces. Therefore, for categorical HPs
                # identical categorical HP value ranges are required for the warm start and the following optimization
                warmstart_params[this_param] = trial.suggest_categorical(name=this_param,
                                                                         choices=
                                                                         list(self.hp_space[i].categories))

            # Real HPs: Set the upper and lower bound to the default / warmstart value to leave no choice
            elif type(self.hp_space[i]) == skopt.space.space.Real:
                warmstart_params[this_param] = trial.suggest_float(name=this_param,
                                                                   low=default_params[this_param],
                                                                   high=default_params[this_param])
            else:
                raise Exception('The skopt HP-space could not be converted correctly!')

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR' or \
                self.ml_algorithm == 'AdaBoostRegressor' or self.ml_algorithm == 'DecisionTreeRegressor':
            eval_func = self.train_evaluate_scikit_regressor

        elif self.ml_algorithm == 'KerasRegressor':
            eval_func = self.train_evaluate_keras_regressor

        elif self.ml_algorithm == 'XGBoostRegressor':
            eval_func = self.train_evaluate_xgboost_regressor

        else:
            raise Exception('Unknown ML-algorithm!')

        # Compute the validation loss for the warmstart HP-configuration
        warmstart_loss = eval_func(params=warmstart_params)

        return warmstart_loss
