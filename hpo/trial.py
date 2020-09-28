import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from xgboost import XGBRegressor

from hpo.optuna_optimizer import OptunaOptimizer
from hpo.skopt_optimizer import SkoptOptimizer
from hpo.hpbandster_optimizer import HpbandsterOptimizer
from hpo.robo_optimizer import RoboOptimizer
from hpo.hyperopt_optimizer import HyperoptOptimizer
from hpo.results import TrialResult, MetricsResult
from hpo.hpo_metrics import area_under_curve


class Trial:
    def __init__(self, hp_space: list, ml_algorithm: str, optimization_schedule: list, metric,
                 n_runs: int, n_func_evals: int, n_workers: int,
                 x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series):
        self.hp_space = hp_space
        self.ml_algorithm = ml_algorithm
        self.optimization_schedule = optimization_schedule
        self.metric = metric
        self.n_runs = n_runs
        self.n_func_evals = n_func_evals
        self.n_workers = n_workers
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        # Attribute for CPU / GPU selection required

    def run(self):
        """
        Run the hyperparameter optimization according to the optimization schedule.
        :return: trial_results_dict: dict
            Contains the optimization results of this trial
        """

        # Initialize a dictionary for saving the trial results
        trial_results_dict = {}

        # Process the optimization schedule -> Iterate over the tuples (hpo_library, hpo_method)
        for opt_tuple in self.optimization_schedule:
            this_hpo_library = opt_tuple[0]
            this_hpo_method = opt_tuple[1]

            # Initialize a DataFrame for saving the trial results
            results_df = pd.DataFrame(columns=['HPO-library', 'HPO-method', 'ML-algorithm', 'run_id', 'random_seed',
                                               'num_of_evaluation', 'losses', 'timestamps'])
            best_configs = ()
            best_losses = []

            # Perform n_runs with varying random seeds
            for i in range(self.n_runs):
                run_id = str(uuid.uuid4())
                this_seed = i  # Random seed for this run

                # Create an optimizer object
                if this_hpo_library == 'skopt':
                    optimizer = SkoptOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                               ml_algorithm=self.ml_algorithm, x_train=self.x_train, x_val=self.x_val,
                                               y_train=self.y_train, y_val=self.y_val, metric=self.metric,
                                               n_func_evals=self.n_func_evals, random_seed=this_seed)

                elif this_hpo_library == 'optuna':
                    optimizer = OptunaOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                ml_algorithm=self.ml_algorithm, x_train=self.x_train, x_val=self.x_val,
                                                y_train=self.y_train, y_val=self.y_val, metric=self.metric,
                                                n_func_evals=self.n_func_evals, random_seed=this_seed)

                elif this_hpo_library == 'hpbandster':
                    optimizer = HpbandsterOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                    ml_algorithm=self.ml_algorithm, x_train=self.x_train,
                                                    x_val=self.x_val, y_train=self.y_train, y_val=self.y_val,
                                                    metric=self.metric, n_func_evals=self.n_func_evals,
                                                    random_seed=this_seed)

                elif this_hpo_library == 'robo':
                    optimizer = RoboOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                              ml_algorithm=self.ml_algorithm, x_train=self.x_train, x_val=self.x_val,
                                              y_train=self.y_train, y_val=self.y_val, metric=self.metric,
                                              n_func_evals=self.n_func_evals, random_seed=this_seed)

                elif this_hpo_library == 'hyperopt':
                    optimizer = HyperoptOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                  ml_algorithm=self.ml_algorithm, x_train=self.x_train,
                                                  x_val=self.x_val,
                                                  y_train=self.y_train, y_val=self.y_val, metric=self.metric,
                                                  n_func_evals=self.n_func_evals, random_seed=this_seed)

                else:
                    raise Exception('Unknown HPO-library!')

                # Start the optimization
                optimization_results = optimizer.optimize()

                # Save the optimization results in a dictionary
                temp_dict = {'HPO-library': [this_hpo_library] * len(optimization_results.losses),
                             'HPO-method': [this_hpo_method] * len(optimization_results.losses),
                             'ML-algorithm': [self.ml_algorithm] * len(optimization_results.losses),
                             'run_id': [run_id] * len(optimization_results.losses),
                             'random_seed': [i] * len(optimization_results.losses),
                             'num_of_evaluation': list(range(1, len(optimization_results.losses) + 1)),
                             'losses': optimization_results.losses,
                             'timestamps': optimization_results.timestamps}

                # Append the optimization results to the result DataFrame of this trial
                this_df = pd.DataFrame.from_dict(data=temp_dict)
                results_df = pd.concat(objs=[results_df, this_df], axis=0)

                # Retrieve the best HP-configuration and the achieved loss
                best_configs = best_configs + (optimization_results.best_configuration,)
                best_losses.append(optimization_results.best_loss)

            for i in range(len(best_losses)):
                if i == 0:
                    best_loss = best_losses[i]
                    idx_best = i
                elif best_losses[i] < best_loss:
                    best_loss = best_losses[i]
                    idx_best = i

            # Create a TrialResult-object to save the results of this trial
            trial_result_obj = TrialResult(trial_result_df=results_df, best_trial_configuration=best_configs[idx_best],
                                           best_trial_loss=best_loss, hpo_library=this_hpo_library,
                                           hpo_method=this_hpo_method)

            # Append the TrialResult-object to the result dictionary
            trial_results_dict[opt_tuple] = trial_result_obj

        return trial_results_dict

    def plot_learning_curve(self, trial_results_dict: dict):
        """
        Plot the learning curves for the HPO-methods that have been evaluated in a trial.
        :param trial_results_dict: dict
            Contains the optimization results of a trial
        :return: fig: matplotlib.figure.Figure
            Learning curves (loss over time)
        """

        # Initialize the plot figure
        fig, ax = plt.subplots()
        mean_lines = []

        # Iterate over each optimization tuples (hpo-library, hpo-method)
        for opt_tuple in trial_results_dict.keys():

            this_df = trial_results_dict[opt_tuple].trial_result_df
            unique_ids = this_df['run_id'].unique()  # Unique id of each optimization run

            n_cols = len(unique_ids)
            n_rows = 0

            # Find the maximum number of function evaluations over all runs of this tuning tuple
            for uniq in unique_ids:
                num_of_evals = len(this_df.loc[this_df['run_id'] == uniq]['num_of_evaluation'])
                if num_of_evals > n_rows:
                    n_rows = num_of_evals

            # n_rows = int(len(this_df['num_of_evaluation']) / n_cols)
            best_losses = np.zeros(shape=(n_rows, n_cols))
            timestamps = np.zeros(shape=(n_rows, n_cols))

            # Iterate over all runs (with varying random seeds)
            for j in range(n_cols):
                this_subframe = this_df.loc[this_df['run_id'] == unique_ids[j]]
                this_subframe = this_subframe.sort_values(by=['num_of_evaluation'], ascending=True, inplace=False)

                # Iterate over all function evaluations
                for i in range(n_rows):

                    # Append timestamps and the descending loss values (learning curves)
                    try:
                        timestamps[i, j] = this_subframe['timestamps'][i]

                        if i == 0:
                            best_losses[i, j] = this_subframe['losses'][i]

                        elif this_subframe['losses'][i] < best_losses[i - 1, j]:
                            best_losses[i, j] = this_subframe['losses'][i]

                        else:
                            best_losses[i, j] = best_losses[i - 1, j]

                    except:
                        timestamps[i, j] = float('nan')
                        best_losses[i, j] = float('nan')

            # Compute the average loss over all runs
            mean_trace_desc = np.nanmean(best_losses, axis=1)

            # 25% and 75% loss quantile for each point (function evaluation)
            quant25_trace_desc = np.nanquantile(best_losses, q=.25, axis=1)
            quant75_trace_desc = np.nanquantile(best_losses, q=.75, axis=1)

            # Compute average timestamps
            mean_timestamps = np.nanmean(timestamps, axis=1)

            # Plot the mean loss over time
            mean_line = ax.plot(mean_timestamps, mean_trace_desc)
            mean_lines.append(mean_line[0])

            # Colored area to visualize the inter-quantile area
            ax.fill_between(x=mean_timestamps, y1=quant25_trace_desc,
                            y2=quant75_trace_desc, alpha=0.2)

        # Add a horizontal line for the default hyperparameter configuration of the ML-algorithm (baseline)
        baseline_loss = self.get_baseline_loss()
        baseline = ax.hlines(baseline_loss, xmin=min(mean_timestamps), xmax=max(mean_timestamps), linestyles='dashed',
                             colors='m')

        # Formatting of the plot
        plt.xlabel('Wall clock time [s]')
        plt.ylabel('Loss')
        plt.yscale('log')
        mean_lines.append(baseline)
        labels = [this_tuple[1] for this_tuple in trial_results_dict.keys()]
        labels.append('Default HPs')
        plt.legend(mean_lines, labels, loc='upper right')

        return fig

    def get_best_trial_result(self, trial_results_dict: dict) -> dict:
        """
        Determine the best trial result according to the validation loss.
        :param trial_results_dict:
            Contains the optimization results of a trial
        :return: out_dict: dictionary
            Contains the best results of this trial
        """

        # Iterate over each optimization tuple (hpo-library, hpo-method) and determine the result that minimizes
        # the loss
        for i in range(len(trial_results_dict.keys())):
            this_opt_tuple = list(trial_results_dict.keys())[i]

            if i == 0:
                best_loss = trial_results_dict[this_opt_tuple].best_loss
                best_configuration = trial_results_dict[this_opt_tuple].best_trial_configuration
                best_library = trial_results_dict[this_opt_tuple].hpo_library
                best_method = trial_results_dict[this_opt_tuple].hpo_method

            elif trial_results_dict[this_opt_tuple].best_loss < best_loss:
                best_loss = trial_results_dict[this_opt_tuple].best_loss
                best_configuration = trial_results_dict[this_opt_tuple].best_trial_configuration
                best_library = trial_results_dict[this_opt_tuple].hpo_library
                best_method = trial_results_dict[this_opt_tuple].hpo_method

        out_dict = {'ML-algorithm': self.ml_algorithm, 'HPO-method': best_method, 'HPO-library': best_library,
                    'HP-configuration': best_configuration,
                    'Loss': best_loss}
        return out_dict

    def get_metrics(self, trial_results_dict: dict):
        """

        :param trial_results_dict:
        :return: metrics: dict
            Dictionary that contains a dictionary with the computed metrics for each optimization tuple.
        """

        metrics = {}
        baseline_loss = self.get_baseline_loss()

        for opt_tuple in trial_results_dict.keys():

            this_df = trial_results_dict[opt_tuple].trial_result_df
            unique_ids = this_df['run_id'].unique()  # Unique id of each optimization run

            n_cols = len(unique_ids)
            n_rows = 0

            # Find the maximum number of function evaluations over all runs of this tuning tuple
            for uniq in unique_ids:
                num_of_evals = len(this_df.loc[this_df['run_id'] == uniq]['num_of_evaluation'])
                if num_of_evals > n_rows:
                    n_rows = num_of_evals

            # n_rows = int(len(this_df['num_of_evaluation']) / n_cols)
            best_losses = np.zeros(shape=(n_rows, n_cols))
            timestamps = np.zeros(shape=(n_rows, n_cols))

            # Iterate over all runs (with varying random seeds)
            for j in range(n_cols):
                this_subframe = this_df.loc[this_df['run_id'] == unique_ids[j]]
                this_subframe = this_subframe.sort_values(by=['num_of_evaluation'], ascending=True, inplace=False)

                # Iterate over all function evaluations
                for i in range(n_rows):

                    # Append timestamps and the descending loss values (learning curves)
                    try:
                        timestamps[i, j] = this_subframe['timestamps'][i]

                        if i == 0:
                            best_losses[i, j] = this_subframe['losses'][i]

                        elif this_subframe['losses'][i] < best_losses[i - 1, j]:
                            best_losses[i, j] = this_subframe['losses'][i]

                        else:
                            best_losses[i, j] = best_losses[i - 1, j]

                    except:
                        timestamps[i, j] = float('nan')
                        best_losses[i, j] = float('nan')

            # Compute the average loss over all runs
            mean_trace_desc = np.nanmean(best_losses, axis=1)

            # Compute average timestamps
            mean_timestamps = np.nanmean(timestamps, axis=1)

            # ANYTIME PERFORMANCE
            # 1. Wall clock time required to outperform the default configuration
            time_outperform_default = float('inf')
            for eval_num in range(len(mean_trace_desc)):
                if mean_trace_desc[eval_num] < baseline_loss:
                    time_outperform_default = mean_timestamps[eval_num]
                    break

            # 2. Area under curve (AUC)
            auc = area_under_curve(list(mean_trace_desc), lower_bound=0.0)

            # FINAL PERFORMANCE
            # 3. Mean loss of the best configuration
            best_mean_loss = min(mean_trace_desc)

            # 4. Loss ratio (loss of best config. / loss of default config.)
            loss_ratio = baseline_loss / best_mean_loss

            # ROBUSTNESS
            # 5. Standard dev. of the loss of the best found configuration
            std_dev_best_loss = np.nanstd(best_losses, axis=1)
            std_dev_best_loss = std_dev_best_loss[-1]

            # 6. Total number of crashes during the optimization
            # tbd

            # USABILITY
            # 7. Wall clock time to find the best configuration
            for eval_num in range(len(mean_trace_desc)):
                if mean_trace_desc[eval_num] <= best_mean_loss:
                    best_idx = eval_num
                    break
            time_best_config = mean_timestamps[best_idx]

            # 8. Number of function evaluations to find the best configuration
            evals_for_best_config = best_idx + 1

            # Pass the computed metrics to a MetricsResult-object
            metrics_object = MetricsResult(time_outperform_default=time_outperform_default,
                                           area_under_curve=auc,
                                           best_mean_loss=best_mean_loss,
                                           loss_ratio=loss_ratio,
                                           std_dev_best_loss=std_dev_best_loss,
                                           time_best_config=time_best_config,
                                           evals_for_best_config=evals_for_best_config)

            # Assign the MetricsResult-object to a dictionary
            metrics[opt_tuple] = metrics_object

        return metrics

    def get_baseline_loss(self):
        """
        Computes the loss for the default hyperparameter configuration of the ML-algorithm (baseline).
        :return:
        baseline_loss: float
            Validation loss of the baseline HP-configuration
        """
        if self.ml_algorithm == 'RandomForestRegressor':
            model = RandomForestRegressor()
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)

        elif self.ml_algorithm == 'KerasRegressor':
            # >>> What are default parameters for a keras model?
            # Baseline regression model from: https://www.tensorflow.org/tutorials/keras/regression#full_model
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(input_shape=len(self.x_train.keys())))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(1))

            model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001))

            model.fit(self.x_train, self.y_train, epochs=100, validation_data=(self.x_val, self.y_val), verbose=0)

            y_pred = model.predict(self.x_val)

        elif self.ml_algorithm == 'XGBoostRegressor':
            model = XGBRegressor()
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)

        else:
            raise Exception('Unknown ML-algorithm!')

        # Add remaining ML-algorithms here

        baseline_loss = self.metric(self.y_val, y_pred)

        return baseline_loss
