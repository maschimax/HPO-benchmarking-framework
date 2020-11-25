import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import math
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import KFold, train_test_split
from tensorflow import keras
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
import functools

from hpo_framework.optuna_optimizer import OptunaOptimizer
from hpo_framework.skopt_optimizer import SkoptOptimizer
from hpo_framework.hpbandster_optimizer import HpbandsterOptimizer
from hpo_framework.robo_optimizer import RoboOptimizer
from hpo_framework.hyperopt_optimizer import HyperoptOptimizer
from hpo_framework.results import TrialResult, MetricsResult
from hpo_framework.hpo_metrics import area_under_curve
from hpo_framework.hp_spaces import warmstart_keras
from hpo_framework.lr_schedules import fix, exponential, cosine


class Trial:
    def __init__(self, hp_space: list, ml_algorithm: str, optimization_schedule: list, metric,
                 n_runs: int, n_func_evals: int, n_workers: int,
                 x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, val_baseline=0.0,
                 test_baseline=0.0, do_warmstart='No', optimizer=None, gpu=False, cross_val=False):
        self.hp_space = hp_space
        self.ml_algorithm = ml_algorithm
        self.optimization_schedule = optimization_schedule
        self.metric = metric
        self.n_runs = n_runs
        self.n_func_evals = n_func_evals
        self.n_workers = n_workers
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.val_baseline = val_baseline  # Cross validation based baseline (split of training data set)
        self.test_baseline = test_baseline  # Full training based baseline (performance evaluation on test data set)
        self.do_warmstart = do_warmstart
        self.optimizer = optimizer
        self.gpu = gpu
        self.cross_val = cross_val  # Apply cross validation (yes / no)

    def run(self):
        """
        Run the hyperparameter optimization according to the optimization schedule.
        :return: trial_results_dict: dict
            Contains the optimization results of this trial
        """

        # Unique trial ID
        trial_id = str(uuid.uuid4())

        # Initialize a dictionary for saving the trial results
        trial_results_dict = {}

        # Process the optimization schedule -> Iterate over the tuples (hpo_library, hpo_method)
        for opt_tuple in self.optimization_schedule:
            this_hpo_library = opt_tuple[0]
            this_hpo_method = opt_tuple[1]

            # Initialize a DataFrame for saving the trial results
            results_df = pd.DataFrame(
                columns=['Trial-ID', 'HPO-library', 'HPO-method', 'ML-algorithm', 'Run-ID', 'random_seed',
                         'eval_count', 'val_losses', 'val_baseline', 'test_loss [best config.]', 'test_baseline',
                         'timestamps', 'configurations', 'run_successful', 'warmstart', 'runs', 'evaluations',
                         'workers', 'GPU', 'budget [%]', '# training instances', '# training features',
                         '# test instances', '# test features'])

            best_configs = ()
            best_val_losses = []
            test_losses = []

            # Perform n_runs with varying random seeds
            for i in range(self.n_runs):
                run_id = str(uuid.uuid4())
                this_seed = i  # Random seed for this run

                # Create an optimizer object
                if this_hpo_library == 'skopt':
                    optimizer = SkoptOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                               ml_algorithm=self.ml_algorithm, x_train=self.x_train, x_test=self.x_test,
                                               y_train=self.y_train, y_test=self.y_test, metric=self.metric,
                                               n_func_evals=self.n_func_evals, random_seed=this_seed,
                                               n_workers=self.n_workers, do_warmstart=self.do_warmstart,
                                               cross_val=self.cross_val)

                elif this_hpo_library == 'optuna':
                    optimizer = OptunaOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                ml_algorithm=self.ml_algorithm, x_train=self.x_train,
                                                x_test=self.x_test, y_train=self.y_train, y_test=self.y_test,
                                                metric=self.metric, n_func_evals=self.n_func_evals,
                                                random_seed=this_seed, n_workers=self.n_workers,
                                                do_warmstart=self.do_warmstart,
                                                cross_val=self.cross_val)

                elif this_hpo_library == 'hpbandster':
                    optimizer = HpbandsterOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                    ml_algorithm=self.ml_algorithm, x_train=self.x_train,
                                                    x_test=self.x_test, y_train=self.y_train, y_test=self.y_test,
                                                    metric=self.metric, n_func_evals=self.n_func_evals,
                                                    random_seed=this_seed, n_workers=self.n_workers,
                                                    do_warmstart=self.do_warmstart,
                                                    cross_val=self.cross_val)

                elif this_hpo_library == 'robo':
                    optimizer = RoboOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                              ml_algorithm=self.ml_algorithm, x_train=self.x_train, x_test=self.x_test,
                                              y_train=self.y_train, y_test=self.y_test, metric=self.metric,
                                              n_func_evals=self.n_func_evals, random_seed=this_seed,
                                              n_workers=self.n_workers, do_warmstart=self.do_warmstart,
                                              cross_val=self.cross_val)

                elif this_hpo_library == 'hyperopt':
                    optimizer = HyperoptOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                  ml_algorithm=self.ml_algorithm, x_train=self.x_train,
                                                  x_test=self.x_test, y_train=self.y_train, y_test=self.y_test,
                                                  metric=self.metric, n_func_evals=self.n_func_evals,
                                                  random_seed=this_seed, n_workers=self.n_workers,
                                                  cross_val=self.cross_val)

                else:
                    raise Exception('Unknown HPO-library!')

                self.optimizer = optimizer

                # Start the optimization
                optimization_results = optimizer.optimize()

                # Check whether a validation baseline has already been calculated
                if self.val_baseline == 0.0:
                    # Compute a new baseline
                    val_baseline_loss = self.get_baseline(cv_mode=True, test_mode=False)
                    self.val_baseline = val_baseline_loss
                else:
                    val_baseline_loss = self.val_baseline

                # Check whether a test baseline has already been calculated
                if self.test_baseline == 0.0:
                    # Compute a new baseline
                    test_baseline_loss = self.get_baseline(cv_mode=False, test_mode=True)
                    self.test_baseline = test_baseline_loss
                else:
                    test_baseline_loss = self.test_baseline

                # Save the optimization results in a dictionary
                temp_dict = {'Trial-ID': [trial_id] * len(optimization_results.losses),
                             'HPO-library': [this_hpo_library] * len(optimization_results.losses),
                             'HPO-method': [this_hpo_method] * len(optimization_results.losses),
                             'ML-algorithm': [self.ml_algorithm] * len(optimization_results.losses),
                             'Run-ID': [run_id] * len(optimization_results.losses),
                             'random_seed': [i] * len(optimization_results.losses),
                             'eval_count': list(range(1, len(optimization_results.losses) + 1)),
                             'val_losses': optimization_results.losses,
                             'val_baseline': [val_baseline_loss] * len(optimization_results.losses),
                             'test_loss [best config.]': [optimization_results.test_loss] * len(
                                 optimization_results.losses),
                             'test_baseline': [test_baseline_loss] * len(optimization_results.losses),
                             'timestamps': optimization_results.timestamps,
                             'configurations': optimization_results.configurations,
                             'run_successful': optimization_results.successful,
                             'warmstart': optimization_results.did_warmstart,
                             'runs': [self.n_runs] * len(optimization_results.losses),
                             'evaluations': [self.n_func_evals] * len(optimization_results.losses),
                             'workers': [self.n_workers] * len(optimization_results.losses),
                             'GPU': self.gpu,
                             'budget [%]': optimization_results.budget,
                             '# training instances': [len(self.x_train)] * len(optimization_results.losses),
                             '# training features': [len(self.x_train.columns)] * len(optimization_results.losses),
                             '# test instances': [len(self.x_test)] * len(optimization_results.losses),
                             '# test features': [len(self.x_test.columns)] * len(optimization_results.losses)}

                # Append the optimization results to the result DataFrame of this trial
                this_df = pd.DataFrame.from_dict(data=temp_dict)
                results_df = pd.concat(objs=[results_df, this_df], axis=0)

                # Retrieve the best HP-configuration and the achieved loss
                best_configs = best_configs + (optimization_results.best_configuration,)
                best_val_losses.append(optimization_results.best_val_loss)
                test_losses.append(optimization_results.test_loss)

            # Iterate over the runs and find the best configuration and its validation and test loss
            for i in range(len(best_val_losses)):
                if i == 0:
                    best_val_loss = best_val_losses[i]
                    idx_best = i
                elif best_val_losses[i] < best_val_loss:
                    best_val_loss = best_val_losses[i]
                    idx_best = i

            # Best test loss of all runs for this HPO method
            best_test_loss = test_losses[idx_best]

            # Create a TrialResult-object to save the results of this trial
            trial_result_obj = TrialResult(trial_result_df=results_df, best_trial_configuration=best_configs[idx_best],
                                           best_val_loss=best_val_loss, best_test_loss=best_test_loss,
                                           hpo_library=this_hpo_library, hpo_method=this_hpo_method,
                                           did_warmstart=optimization_results.did_warmstart)

            # Append the TrialResult-object to the result dictionary
            trial_results_dict[opt_tuple] = trial_result_obj

        return trial_results_dict

    def plot_learning_curve(self, trial_results_dict: dict):
        """
        Plot the learning curves for the HPO-methods that have been evaluated in a trial.
        :param trial_results_dict: dict
            Contains the optimization results of a trial.
        :return: fig: matplotlib.figure.Figure
            Learning curves (loss over time)
        """

        # Initialize the plot figure
        fig, ax = plt.subplots()
        mean_lines = []
        max_time = 0  # necessary to limit the length of the baseline curve (default configuration)

        # Iterate over each optimization tuples (hpo-library, hpo-method)
        for opt_tuple in trial_results_dict.keys():

            this_df = trial_results_dict[opt_tuple].trial_result_df
            unique_ids = this_df['Run-ID'].unique()  # Unique id of each optimization run

            n_cols = len(unique_ids)
            n_rows = 0

            # Find the maximum number of function evaluations over all runs of this tuning tuple
            for uniq in unique_ids:
                num_of_evals = len(this_df.loc[this_df['Run-ID'] == uniq]['eval_count'])
                if num_of_evals > n_rows:
                    n_rows = num_of_evals

            # n_rows = int(len(this_df['eval_count']) / n_cols)
            best_losses = np.zeros(shape=(n_rows, n_cols))
            timestamps = np.zeros(shape=(n_rows, n_cols))

            # Iterate over all runs (with varying random seeds)
            for j in range(n_cols):
                this_subframe = this_df.loc[this_df['Run-ID'] == unique_ids[j]]
                this_subframe = this_subframe.sort_values(by=['eval_count'], ascending=True, inplace=False)

                # Iterate over all function evaluations
                for i in range(n_rows):

                    # Append timestamps and the descending loss values (learning curves)
                    try:
                        timestamps[i, j] = this_subframe['timestamps'][i]

                        if i == 0:
                            best_losses[i, j] = this_subframe['val_losses'][i]

                        elif this_subframe['val_losses'][i] < best_losses[i - 1, j]:
                            best_losses[i, j] = this_subframe['val_losses'][i]

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

            if max(mean_timestamps) > max_time:
                max_time = max(mean_timestamps)

            # Plot the mean loss over time
            mean_line = ax.plot(mean_timestamps, mean_trace_desc)
            mean_lines.append(mean_line[0])

            # Colored area to visualize the inter-quantile area
            ax.fill_between(x=mean_timestamps, y1=quant25_trace_desc,
                            y2=quant75_trace_desc, alpha=0.2)

        # Check whether a validation baseline has already been calculated
        if self.val_baseline == 0.0:
            # Compute a new baseline
            val_baseline_loss = self.get_baseline(cv_mode=True, test_mode=False)
            self.val_baseline = val_baseline_loss
        else:
            val_baseline_loss = self.val_baseline

        # Add a horizontal line for the default hyperparameter configuration of the ML-algorithm (baseline)
        baseline = ax.hlines(val_baseline_loss, xmin=0, xmax=max_time, linestyles='dashed',
                             colors='m')

        # Formatting of the plot
        plt.xlabel('Wall clock time [s]')
        plt.ylabel('Validation loss')
        # plt.ylim([0.02, 1.0])
        # plt.xlim([0.05, 1000])
        plt.yscale('log')
        plt.xscale('log')
        # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        # Add a legend
        mean_lines.append(baseline)
        legend_labels = [this_tuple[1] for this_tuple in trial_results_dict.keys()]
        legend_labels.append('Default HPs')
        plt.legend(mean_lines, legend_labels, loc='upper right')

        # Add a title
        font = {'weight': 'semibold',
                'size': 'large'}

        title_label = self.ml_algorithm + " - " + str(self.n_workers) + " worker(s) - " + str(self.n_runs) + " runs"
        plt.title(label=title_label, fontdict=font, loc='center')

        return fig

    @staticmethod
    def plot_hp_space(trial_results_dict: dict):
        """
        Plots a sns.pairplot that visualizes the explored hyperparameter space and highlights the best 5 % of the
        hyperparameter configurations.
        :param trial_results_dict: dict
            Contains the optimization results of a trial.
        :return: plots_dict: dict
            Dictionary that contains a sns.pairplot for each optimization tuple (ML-algorithm, HPO-method).
        """

        # Initialize a dictionary for saving the plots
        plots_dict = {}

        # Iterate over each optimization tuples (hpo-library, hpo-method)
        for opt_tuple in trial_results_dict.keys():

            # Pandas DataFrame containing the optimization results
            this_df = trial_results_dict[opt_tuple].trial_result_df

            # Drop NaN rows from crashed runs
            this_df.dropna(inplace=True)

            if len(this_df['val_losses']) > 0:
                # Strings used in the plot title
                ml_algorithm = this_df.iloc[0]['ML-algorithm']
                hpo_method = opt_tuple[1]
                warmstart = str(this_df.iloc[0]['warmstart'])

                # Sort DataFrame by loss values
                sorted_df = this_df.sort_values(by='val_losses', axis=0, ascending=True, inplace=False)
                sorted_df.reset_index(drop=True, inplace=True)

                # Find the indices of the 5 % best hyperparameter configurations
                n_best_configs = round(.05 * len(sorted_df['val_losses']))
                idx_best_configs = sorted_df.index[:n_best_configs]

                # New column to distinguish the 'best' and the remaining configurations
                sorted_df['Score'] = 'Rest'
                sorted_df.loc[idx_best_configs, 'Score'] = 'Best 5%'

                # Sort by descending losses to ensure that the best configurations are plotted on top
                sorted_df.sort_values(by='val_losses', axis=0, ascending=False, inplace=True)

                # Tuned / Optimized hyperparameters
                hyper_params = list(sorted_df['configurations'].iloc[0].keys())

                # Divide the single column with all hyperparameters (sorted_df) into individual columns for each
                # hyperparameter and assign the evaluated parameter values
                sns_dict = {}
                for param in hyper_params:
                    sns_dict[param] = []

                sns_dict['Score'] = list(sorted_df['Score'])

                for _, row in sorted_df.iterrows():
                    this_config_dict = row['configurations']

                    for param in hyper_params:
                        sns_dict[param].append(this_config_dict[param])

                # Convert dictionary into a pd.DataFrame
                sns_df = pd.DataFrame.from_dict(data=sns_dict)

                # Set ipt colors
                ipt_colors = ['#5cbaa4', '#0062a5']
                sns.set_palette(ipt_colors)

                # Plot seaborn pairplot
                space_plot = sns.pairplot(data=sns_df, hue='Score', hue_order=['Rest', 'Best 5%'])

                # Set plot title
                title_str = ml_algorithm + ' - ' + hpo_method + ' - Warmstart: ' + warmstart
                space_plot.fig.suptitle(title_str, y=1.05, fontweight='semibold')

                # Assign the seaborn pairplot to the dictionary
                plots_dict[opt_tuple] = space_plot

        return plots_dict

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
                best_val_loss = trial_results_dict[this_opt_tuple].best_val_loss
                best_test_loss = trial_results_dict[this_opt_tuple].best_test_loss
                best_configuration = trial_results_dict[this_opt_tuple].best_trial_configuration
                best_library = trial_results_dict[this_opt_tuple].hpo_library
                best_method = trial_results_dict[this_opt_tuple].hpo_method

            elif trial_results_dict[this_opt_tuple].best_val_loss < best_val_loss:
                best_val_loss = trial_results_dict[this_opt_tuple].best_val_loss
                best_test_loss = trial_results_dict[this_opt_tuple].best_test_loss
                best_configuration = trial_results_dict[this_opt_tuple].best_trial_configuration
                best_library = trial_results_dict[this_opt_tuple].hpo_library
                best_method = trial_results_dict[this_opt_tuple].hpo_method

        out_dict = {'ML-algorithm': self.ml_algorithm, 'HPO-method': best_method, 'HPO-library': best_library,
                    'HP-configuration': best_configuration,
                    'Validation loss': best_val_loss, 'Test loss': best_test_loss}
        return out_dict

    def get_metrics(self, trial_results_dict: dict):
        """

        :param trial_results_dict:
        :return: metrics: dict, metrics: pd.DataFrame
            Dictionary that contains a dictionary with the computed metrics for each optimization tuple.
            Pandas DataFrame that contains the computed metrics.
        """

        metrics = {}
        cols = ['Trial-ID', 'HPO-library', 'HPO-method', 'ML-algorithm', 'Runs', 'Evaluations', 'Workers', 'GPU',
                'Warmstart', 'Wall clock time [s]', 't outperform default [s]', 'Mean (final validation loss)',
                'Validation baseline', 'Area under curve (AUC)', 'Mean (final test loss)',
                'Test loss ratio (default / best)', 'Test baseline', 'Interquartile range (final test loss)',
                't best configuration [s]', 'Generalization error', 'Evaluations for best configuration',
                'Crashes', '# training instances', '# training features', '# test instances', '# test features']

        metrics_df = pd.DataFrame(columns=cols)

        # Check whether a validation baseline has already been calculated
        if self.val_baseline == 0.0:
            # Compute a new baseline
            val_baseline = self.get_baseline(cv_mode=True, test_mode=False)
            self.val_baseline = val_baseline
        else:
            val_baseline = self.val_baseline

        # Row index for pandas DataFrame
        idx = 1

        for opt_tuple in trial_results_dict.keys():

            this_df = trial_results_dict[opt_tuple].trial_result_df
            unique_ids = this_df['Run-ID'].unique()  # Unique id of each optimization run

            # Flag indicates, whether a warmstart of the HPO-method was performed successfully
            did_warmstart = trial_results_dict[opt_tuple].did_warmstart

            n_cols = len(unique_ids)
            n_rows = 0

            # Find the maximum number of function evaluations over all runs of this tuning tuple
            for uniq in unique_ids:
                num_of_evals = len(this_df.loc[this_df['Run-ID'] == uniq]['eval_count'])
                if num_of_evals > n_rows:
                    n_rows = num_of_evals

            # n_rows = int(len(this_df['num_of_evaluation']) / n_cols)
            best_val_losses = np.zeros(shape=(n_rows, n_cols))
            timestamps = np.zeros(shape=(n_rows, n_cols))
            best_test_losses = np.zeros(shape=(1, n_cols))

            # Count the number of algorithm crashes that occurred during optimization
            number_of_crashes_this_algo = 0

            # Iterate over all runs (with varying random seeds)
            for j in range(n_cols):
                this_subframe = this_df.loc[this_df['Run-ID'] == unique_ids[j]]
                this_subframe = this_subframe.sort_values(by=['eval_count'], ascending=True, inplace=False)

                best_test_losses[0, j] = this_subframe['test_loss [best config.]'][0]

                # Check, whether this run was completed successfully
                if not all(this_subframe['run_successful']):
                    number_of_crashes_this_algo = number_of_crashes_this_algo + 1

                # Iterate over all function evaluations
                for i in range(n_rows):

                    # Append timestamps and the descending loss values (learning curves)
                    try:
                        timestamps[i, j] = this_subframe['timestamps'][i]

                        if i == 0:
                            best_val_losses[i, j] = this_subframe['val_losses'][i]

                        elif this_subframe['val_losses'][i] < best_val_losses[i - 1, j]:
                            best_val_losses[i, j] = this_subframe['val_losses'][i]

                        else:
                            best_val_losses[i, j] = best_val_losses[i - 1, j]

                    except:
                        timestamps[i, j] = float('nan')
                        best_val_losses[i, j] = float('nan')

            # Compute the average validation loss for each run
            mean_trace_desc = np.nanmean(best_val_losses, axis=1)

            # Compute average timestamps
            mean_timestamps = np.nanmean(timestamps, axis=1)

            # Wall clock time
            wall_clock_time = max(mean_timestamps)

            # ANYTIME PERFORMANCE
            # 1. Wall clock time required to outperform the default configuration (on the validation set)
            time_outperform_default = float('inf')
            for eval_num in range(len(mean_trace_desc)):
                if mean_trace_desc[eval_num] < val_baseline:
                    time_outperform_default = mean_timestamps[eval_num]
                    break

            # 2. Area under curve (AUC)
            auc = area_under_curve(list(mean_trace_desc), lower_bound=0.0)

            # FINAL PERFORMANCE
            # 3.1 Mean validation loss of the best configuration
            best_mean_val_loss = min(mean_trace_desc)

            # 3.2 Mean test loss of the best configuration (full training)
            mean_test_loss = np.mean(best_test_losses)

            # 3.3 Generalization error
            generalization_err = mean_test_loss - best_mean_val_loss

            # 4. Loss ratio (test loss of default config. / test loss of best found config.)
            # Check whether a test baseline has already been calculated
            if self.test_baseline == 0.0:
                # Compute a new baseline
                test_baseline_loss = self.get_baseline(cv_mode=False, test_mode=True)
                self.test_baseline = test_baseline_loss
            else:
                test_baseline_loss = self.test_baseline

            loss_ratio = test_baseline_loss / mean_test_loss

            # ROBUSTNESS
            # 5. Interquantile range of the test loss of the best found configuration
            quant75 = np.nanquantile(best_test_losses, q=.75, axis=1)
            quant25 = np.nanquantile(best_test_losses, q=.25, axis=1)
            interq_range = (quant75 - quant25)[-1]

            # 6. Total number of crashes during the optimization (for each HPO-method)
            # number_of_crashes_this_algo

            # USABILITY
            if math.isnan(best_mean_val_loss):
                # Only crashed runs for this HPO-method
                best_idx = float('nan')
                time_best_config = float('nan')
                evals_for_best_config = float('nan')

            else:
                for eval_num in range(len(mean_trace_desc)):
                    if mean_trace_desc[eval_num] <= best_mean_val_loss:
                        best_idx = eval_num  # index of the first evaluation, that reaches the best loss
                        break

                # 7. Wall clock time to find the best configuration
                time_best_config = mean_timestamps[best_idx]

                # 8. Number of function evaluations to find the best configuration
                evals_for_best_config = best_idx + 1

            # Pass the computed metrics to a MetricsResult-object
            metrics_object = MetricsResult(wall_clock_time=wall_clock_time,
                                           time_outperform_default=time_outperform_default,
                                           area_under_curve=auc,
                                           mean_test_loss=mean_test_loss,
                                           loss_ratio=loss_ratio,
                                           interquantile_range=interq_range,
                                           time_best_config=time_best_config,
                                           evals_for_best_config=evals_for_best_config,
                                           number_of_crashes=number_of_crashes_this_algo)

            # Assign the MetricsResult-object to a dictionary
            metrics[opt_tuple] = metrics_object

            # ID of this Trial
            trial_id = this_df['Trial-ID'].unique()[0]

            # Dictionary with new metrics
            metrics_dict = {'Trial-ID': trial_id,
                            'idx': [idx],
                            'HPO-library': opt_tuple[0],
                            'HPO-method': opt_tuple[1],
                            'ML-algorithm': self.ml_algorithm,
                            'Runs': self.n_runs,
                            'Evaluations': self.n_func_evals,
                            'Workers': self.n_workers,
                            'GPU': self.gpu,
                            'Warmstart': did_warmstart,
                            'Wall clock time [s]': wall_clock_time,
                            't outperform default [s]': time_outperform_default,
                            'Mean (final validation loss)': best_mean_val_loss,
                            'Validation baseline': val_baseline,
                            'Area under curve (AUC)': auc,
                            'Mean (final test loss)': mean_test_loss,
                            'Test loss ratio (default / best)': loss_ratio,
                            'Test baseline': test_baseline_loss,
                            'Interquartile range (final test loss)': interq_range,
                            't best configuration [s]': time_best_config,
                            'Generalization error': generalization_err,
                            'Evaluations for best configuration': evals_for_best_config,
                            'Crashes': number_of_crashes_this_algo,
                            '# training instances': len(self.x_train),
                            '# training features': len(self.x_train.columns),
                            '# test instances': len(self.x_test),
                            '# test features': len(self.x_test.columns)}

            # Create pandas DataFrame from dictionary
            this_metrics_df = pd.DataFrame.from_dict(data=metrics_dict)
            this_metrics_df.set_index(keys='idx', drop=True, inplace=True)

            # Append the new metrics / results to the whole metrics DataFrame
            metrics_df = pd.concat(objs=[metrics_df, this_metrics_df], axis=0)

            idx = idx + 1

        return metrics, metrics_df

    def get_baseline(self, cv_mode=True, test_mode=False):
        """
        Computes a loss baseline for the ML-algorithm based on its default hyperparameter configuration
        (either cross validation loss or test loss after full training)
        :param cv_mode: bool
            Flag that indicates, whether to perform cross validation or simple validation
        :param test_mode: bool
            Flag that indicates, whether to compute the loss on the test set or not
        :return:
        baseline: float
             Loss of the baseline HP-configuration.
        """

        # Create K-Folds cross validator
        kf = KFold(n_splits=5)
        cv_baselines = []
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

            if self.ml_algorithm == 'RandomForestRegressor':
                model = RandomForestRegressor(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'RandomForestClassifier':
                model = RandomForestClassifier(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'SVR':
                model = SVR(cache_size=500)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'SVC':
                model = SVC(random_state=0, cache_size=500)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'AdaBoostRegressor':
                model = AdaBoostRegressor(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'AdaBoostClassifier':
                model = AdaBoostClassifier(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'DecisionTreeRegressor':
                model = DecisionTreeRegressor(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'DecisionTreeClassifier':
                model = DecisionTreeClassifier(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'LinearRegression':
                model = LinearRegression()
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'KNNRegressor':
                model = KNeighborsRegressor()
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'KNNClassifier':
                model = KNeighborsClassifier()
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'LogisticRegression':
                model = LogisticRegression()
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'NaiveBayes':
                model = GaussianNB()
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'MLPRegressor':
                model = MLPRegressor(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'MLPClassifier':
                model = MLPClassifier(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'ElasticNet':
                model = ElasticNet(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'KerasRegressor' or self.ml_algorithm == 'KerasClassifier':

                # Use the warmstart configuration to create a baseline for Keras models

                epochs = 100

                # Initialize the neural network
                model = keras.Sequential()

                # Add input layer
                model.add(keras.layers.InputLayer(input_shape=len(x_train_cv.keys())))

                # Add first hidden layer
                if warmstart_keras['hidden_layer1_size'] > 0:
                    model.add(
                        keras.layers.Dense(warmstart_keras['hidden_layer1_size'],
                                           activation=warmstart_keras['hidden_layer1_activation']))
                    model.add(keras.layers.Dropout(warmstart_keras['dropout1']))

                # Add second hidden layer
                if warmstart_keras['hidden_layer2_size'] > 0:
                    model.add(
                        keras.layers.Dense(warmstart_keras['hidden_layer2_size'],
                                           activation=warmstart_keras['hidden_layer2_activation']))
                    model.add(keras.layers.Dropout(warmstart_keras['dropout2']))

                # Add output layer
                if self.ml_algorithm == 'KerasRegressor':

                    model.add(keras.layers.Dense(1, activation='linear'))

                    # Select optimizer and compile the model
                    adam = keras.optimizers.Adam(learning_rate=warmstart_keras['init_lr'])
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

                        adam = keras.optimizers.Adam(learning_rate=warmstart_keras['init_lr'])
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
                if warmstart_keras["lr_schedule"] == "cosine":
                    schedule = functools.partial(cosine, initial_lr=warmstart_keras["init_lr"], T_max=epochs)

                elif warmstart_keras["lr_schedule"] == "exponential":
                    schedule = functools.partial(exponential, initial_lr=warmstart_keras["init_lr"], T_max=epochs)

                elif warmstart_keras["lr_schedule"] == "constant":
                    schedule = functools.partial(fix, initial_lr=warmstart_keras["init_lr"])

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
                model.fit(x_train_cv, y_train_cv, epochs=epochs, batch_size=warmstart_keras['batch_size'],
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

            elif self.ml_algorithm == 'XGBoostRegressor':
                model = XGBRegressor(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'XGBoostClassifier':
                model = XGBClassifier(random_state=0)
                model.fit(x_train_cv, y_train_cv)
                y_pred = model.predict(x_val_cv)

            elif self.ml_algorithm == 'LGBMRegressor' or self.ml_algorithm == 'LGBMClassifier':
                # Create lgb datasets
                train_data = lgb.Dataset(x_train_cv, label=y_train_cv)
                valid_data = lgb.Dataset(x_val_cv, label=y_val_cv)

                # Specify the ML task and the random seed
                if self.ml_algorithm == 'LGBMRegressor':
                    # Regression task
                    params = {'objective': 'regression',
                              'seed': 0}

                elif self.ml_algorithm == 'LGBMClassifier':

                    # Determine the number of classes
                    num_classes = int(max(y_train_cv) - min(y_train_cv) + 1)

                    # Binary classification task
                    if num_classes <= 2:
                        params = {'objective': 'binary',
                                  'seed': 0}

                    # Multiclass classification task
                    else:
                        params = {'objective': 'multiclass',  # uses Softmax objective function
                                  'num_class': num_classes,
                                  'seed': 0}

                lgb_clf = lgb.train(params=params, train_set=train_data, valid_sets=[valid_data], verbose_eval=False)

                # Make the prediction
                y_pred = lgb_clf.predict(data=x_val_cv)

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

            else:
                raise Exception('Unknown ML-algorithm!')

            # Add remaining ML-algorithms here

            cv_baselines.append(self.metric(y_val_cv, y_pred))

        if cv_mode:

            # Compute the average cross validation loss
            baseline = np.mean(cv_baselines)

        else:
            baseline = cv_baselines[0]

        return baseline

    @staticmethod
    def train_best_model(trial_results_dict: dict):
        # Method to train the ML model with the best found HP configuration on the whole dataset
        raise NotImplementedError
