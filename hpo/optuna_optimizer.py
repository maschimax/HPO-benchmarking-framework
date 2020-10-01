import optuna
import skopt
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
import time
from multiprocessing import Pool

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


def load_study_and_optimize(st_name, st_storage, n_func_evals, objective_func):
    this_study = optuna.load_study(st_name, st_storage)
    this_study.optimize(objective_func, n_func_evals)
    return

class OptunaOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                 random_seed, n_workers):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                         random_seed, n_workers)

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

        elif self.hpo_method == 'RandomSearch':
            this_optimizer = RandomSampler(seed=self.random_seed)

        else:
            raise Exception('Unknown HPO-method!')

        # Create a study object and specify the optimization direction
        study_name = 'hpo_study'
        study_storage = 'sqlite:///hpo.db'
        study = optuna.create_study(sampler=this_optimizer, direction='minimize',
                                    study_name=study_name, storage=study_storage, load_if_exists=True)

        # Optimize on the predefined n_func_evals and measure the wall clock times
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Start the optimization
        # try:
            # for worker in range(self.n_workers):
            #     worker = optuna.load_study(study_name='hpo_study', storage=study_storage)
            #     worker.optimize(func=self.objective, n_trials=self.n_func_evals)

        with Pool(processes=self.n_workers) as pool:
            pool.apply(func=load_study_and_optimize, args=(study_name, study_name, self.n_func_evals, self.objective))
            pool.close()
            pool.join()

        # study.optimize(func=self.objective, n_trials=self.n_func_evals)
        run_successful = True

        # Algorithm crashed
        # except:
        #     # Add a warning here
        #     run_successful = False

        # If the optimization run was successful, determine the optimization results
        if run_successful:

            for i in range(len(self.times)):
                # Subtract the start time to receive the wall clock time of each function evaluation
                self.times[i] = self.times[i] - start_time
            wall_clock_time = max(self.times)

            # Timestamps
            timestamps = self.times

            # Create a TuningResult-object to store the optimization results
            # Transformation of the results into a TuningResult-Object
            all_trials = study.get_trials()
            best_configuration = study.best_params
            best_loss = study.best_value

            evaluation_ids = []  # Number the evaluations / iterations of this run
            losses = []  # Loss of each iteration
            configurations = ()  # HP-configuration of each iteration
            for i in range(len(all_trials)):
                evaluation_ids.append(all_trials[i].number)
                losses.append(all_trials[i].value)
                configurations = configurations + (all_trials[i].params,)

        # Run not successful (algorithm crashed)
        else:
            evaluation_ids, timestamps, losses, configurations, best_loss, best_configuration, wall_clock_time = \
                self.impute_results_for_crash()

        # Pass the results to a TuningResult-object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=timestamps, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_configuration,
                              wall_clock_time=wall_clock_time, successful=run_successful)

        return result

    def objective(self, trial):
        """
        Objective function: This method converts the given hyperparameters into a dictionary, passes them to the
        ML-model for training and returns the validation loss.
        :param trial: optuna.trial._trial.Trial
            Optuna Trial object, that suggests the next HP-configuration according to the selected HPO-method.
        :return: eval_func(params=dict_params)
            Validation loss for the HP-configuration
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
                                                                               choices=list(self.hp_space[i].categories))

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
