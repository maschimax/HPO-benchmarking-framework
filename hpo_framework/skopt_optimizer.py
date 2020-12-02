from skopt.optimizer import forest_minimize, gp_minimize
import time
import skopt

from hpo_framework.baseoptimizer import BaseOptimizer
from hpo_framework.results import TuningResult


class SkoptOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_test, y_train, y_test, metric, n_func_evals,
                 random_seed, n_workers, do_warmstart, cross_val):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_test, y_train, y_test, metric, n_func_evals,
                         random_seed, n_workers, cross_val)

        self.do_warmstart = do_warmstart

    def optimize(self) -> TuningResult:
        """
        Method performs a hyperparameter optimization run according to the selected HPO-method.
        :return: result: TuningResult
            TuningResult-object that contains the results of this optimization run.
        """

        # Select the specified HPO-tuning method
        if self.hpo_method == 'SMAC':
            # SMAC expects a budget of at least 10 iterations / calls
            this_optimizer = forest_minimize
            this_acq_func = 'EI'

        elif self.hpo_method == 'GPBO':
            this_optimizer = gp_minimize
            this_acq_func = 'EI'

        else:
            raise Exception('Unknown HPO-method!')

        # Use a warmstart configuration?
        if self.do_warmstart == 'Yes':

            try:

                # Initialize a list for saving the warmstart configuration
                warmstart_config = []

                # Retrieve the warmstart hyperparameters for the ML-algorithm
                warmstart_params = self.get_warmstart_configuration()

                # Iterate over all hyperparameters of this ML-algorithm's tuned HP-space and append the default values
                # to the list
                for i in range(len(self.hp_space)):

                    this_param = self.hp_space[i].name
                    this_warmstart_value = warmstart_params[this_param]

                    # For some HPs (e.g. max_depth of RF) the default value is None, although their typical dtype is
                    # different (e.g. int)
                    if this_warmstart_value is None and type(self.hp_space[i]) == skopt.space.space.Integer:
                        # Try to impute these values by the mean value
                        warmstart_config.append(int(0.5 * (self.hp_space[i].low + self.hp_space[i].high)))

                    # If the continuous HP is sampled from the log domain,
                    # transform the warmstart value (log_base**warmstart_value)
                    elif type(self.hp_space[i]) == skopt.space.space.Real and self.hp_space[i].prior == 'log-uniform':

                        warmstart_config.append(self.hp_space[i].base ** this_warmstart_value)

                    else:
                        # Otherwise append the warmstart value (default case)
                        warmstart_config.append(this_warmstart_value)

                # Pass the warmstart configuration as a kwargs dict
                kwargs = {'x0': warmstart_config}

                # Set flag to indicate that a warmstart took place
                did_warmstart = True

            except:
                print('Warmstarting skopt failed!')
                kwargs = {}

                # Set flag to indicate that NO warmstart took place
                did_warmstart = False

        # No warmstart requested
        else:
            kwargs = {}

            # Set flag to indicate that NO warmstart took place
            did_warmstart = False

        # Optimize on the predefined n_func_evals and measure the wall clock times
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Necessary HP space transformation for log sampling to ensure comparability with other HPO libraries
        for i in range(len(self.hp_space)):

            if type(self.hp_space[i]) == skopt.space.space.Real:
                # If the continuous HP is sampled from the log domain,
                # transform the specified lower and upper bounds to the log domain (to keep similar boundaries)
                if self.hp_space[i].prior == 'log-uniform':
                    self.hp_space[i].low = self.hp_space[i].base ** self.hp_space[i].low
                    self.hp_space[i].high = self.hp_space[i].base ** self.hp_space[i].high

        # Start the optimization
        try:
            trial_result = this_optimizer(self.objective, self.hp_space, n_calls=self.n_func_evals,
                                          random_state=self.random_seed, acq_func=this_acq_func,
                                          n_jobs=self.n_workers, verbose=True, n_initial_points=20, **kwargs)

            run_successful = True

        # Algorithm crashed
        except:
            run_successful = False

        # If the optimization run was successful, determine the optimization results
        if run_successful:

            for i in range(len(self.times)):
                # Subtract the start time to receive the wall clock time of each function evaluation
                self.times[i] = self.times[i] - start_time
            wall_clock_time = max(self.times)

            # Timestamps
            timestamps = self.times

            best_val_loss = trial_result.fun

            # Losses (not incumbent losses)
            losses = list(trial_result.func_vals)

            # Determine the best HP-configuration of this run
            best_configuration = {}
            for i in range(len(self.hp_space)):
                best_configuration[self.hp_space[i].name] = trial_result.x[i]

            # Number the evaluations / iterations of this run
            evaluation_ids = list(range(1, len(trial_result.func_vals) + 1))

            # Determine the HP-configuration of each evaluation / iteration
            configurations = ()
            for i in range(len(trial_result.x_iters)):
                this_config = {}
                for j in range(len(self.hp_space)):
                    this_config[self.hp_space[j].name] = trial_result.x_iters[i][j]
                configurations = configurations + (this_config,)

            # Skopt uses full budgets for its HPO methods
            budget = [100.0] * len(losses)

            # Compute the loss on the test set for the best found configuration
            test_loss = self.train_evaluate_ml_model(params=best_configuration, cv_mode=False, test_mode=True)

        # Run not successful (algorithm crashed)
        else:
            evaluation_ids, timestamps, losses, configurations, best_val_loss, best_configuration, wall_clock_time, \
                test_loss, budget = self.impute_results_for_crash()

        # Pass the results to a TuningResult-object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=timestamps, losses=losses,
                              configurations=configurations, best_val_loss=best_val_loss,
                              best_configuration=best_configuration, wall_clock_time=wall_clock_time,
                              test_loss=test_loss, successful=run_successful, did_warmstart=did_warmstart, budget=budget)

        return result

    def objective(self, params):
        """
        Objective function: This method converts the given hyperparameters into a dictionary, passes them to the
        ML-model for training and returns the validation loss.
        :param params: dict
            Hyperparameter configuration that has been selected by the HPO-method for this iteration.
        :return: eval_func(params=dict_params)
            Validation loss for the HP-configuration (params)
        """

        # Convert the hyperparameters into a dictionary to pass them to the ML-model
        dict_params = {}
        for i in range(len(self.hp_space)):
            dict_params[self.hp_space[i].name] = params[i]

        # Compute the validation loss
        val_loss = self.train_evaluate_ml_model(params=dict_params, cv_mode=self.cross_val, test_mode=False)

        return val_loss
