import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import math
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
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
                 x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series, baseline=0.0,
                 do_warmstart='No'):
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
        self.baseline = baseline
        self.do_warmstart = do_warmstart
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
                                               'eval_count', 'losses', 'timestamps', 'configurations',
                                               'run_successful', 'warmstart', 'runs', 'evaluations', 'workers'])
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
                                               n_func_evals=self.n_func_evals, random_seed=this_seed,
                                               n_workers=self.n_workers, do_warmstart=self.do_warmstart)

                elif this_hpo_library == 'optuna':
                    optimizer = OptunaOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                ml_algorithm=self.ml_algorithm, x_train=self.x_train, x_val=self.x_val,
                                                y_train=self.y_train, y_val=self.y_val, metric=self.metric,
                                                n_func_evals=self.n_func_evals, random_seed=this_seed,
                                                n_workers=self.n_workers, do_warmstart=self.do_warmstart)

                elif this_hpo_library == 'hpbandster':
                    optimizer = HpbandsterOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                    ml_algorithm=self.ml_algorithm, x_train=self.x_train,
                                                    x_val=self.x_val, y_train=self.y_train, y_val=self.y_val,
                                                    metric=self.metric, n_func_evals=self.n_func_evals,
                                                    random_seed=this_seed, n_workers=self.n_workers,
                                                    do_warmstart=self.do_warmstart)

                elif this_hpo_library == 'robo':
                    optimizer = RoboOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                              ml_algorithm=self.ml_algorithm, x_train=self.x_train, x_val=self.x_val,
                                              y_train=self.y_train, y_val=self.y_val, metric=self.metric,
                                              n_func_evals=self.n_func_evals, random_seed=this_seed,
                                              n_workers=self.n_workers, do_warmstart=self.do_warmstart)

                elif this_hpo_library == 'hyperopt':
                    optimizer = HyperoptOptimizer(hp_space=self.hp_space, hpo_method=this_hpo_method,
                                                  ml_algorithm=self.ml_algorithm, x_train=self.x_train,
                                                  x_val=self.x_val,
                                                  y_train=self.y_train, y_val=self.y_val, metric=self.metric,
                                                  n_func_evals=self.n_func_evals, random_seed=this_seed,
                                                  n_workers=self.n_workers)

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
                             'eval_count': list(range(1, len(optimization_results.losses) + 1)),
                             'losses': optimization_results.losses,
                             'timestamps': optimization_results.timestamps,
                             'configurations': optimization_results.configurations,
                             'run_successful': optimization_results.successful,
                             'warmstart': optimization_results.did_warmstart,
                             'runs': [self.n_runs] * len(optimization_results.losses),
                             'evaluations': [self.n_func_evals] * len(optimization_results.losses),
                             'workers': [self.n_workers] * len(optimization_results.losses)}

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
                                           hpo_method=this_hpo_method, did_warmstart=optimization_results.did_warmstart)

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
            unique_ids = this_df['run_id'].unique()  # Unique id of each optimization run

            n_cols = len(unique_ids)
            n_rows = 0

            # Find the maximum number of function evaluations over all runs of this tuning tuple
            for uniq in unique_ids:
                num_of_evals = len(this_df.loc[this_df['run_id'] == uniq]['eval_count'])
                if num_of_evals > n_rows:
                    n_rows = num_of_evals

            # n_rows = int(len(this_df['eval_count']) / n_cols)
            best_losses = np.zeros(shape=(n_rows, n_cols))
            timestamps = np.zeros(shape=(n_rows, n_cols))

            # Iterate over all runs (with varying random seeds)
            for j in range(n_cols):
                this_subframe = this_df.loc[this_df['run_id'] == unique_ids[j]]
                this_subframe = this_subframe.sort_values(by=['eval_count'], ascending=True, inplace=False)

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

            if max(mean_timestamps) > max_time:
                max_time = max(mean_timestamps)

            # Plot the mean loss over time
            mean_line = ax.plot(mean_timestamps, mean_trace_desc)
            mean_lines.append(mean_line[0])

            # Colored area to visualize the inter-quantile area
            ax.fill_between(x=mean_timestamps, y1=quant25_trace_desc,
                            y2=quant75_trace_desc, alpha=0.2)

        # Check whether a baseline has already been calculated
        if self.baseline == 0.0:
            # Compute a new baseline
            baseline_loss = self.get_baseline_loss()
            self.baseline = baseline_loss
        else:
            baseline_loss = self.baseline

        # Add a horizontal line for the default hyperparameter configuration of the ML-algorithm (baseline)
        baseline = ax.hlines(baseline_loss, xmin=0, xmax=max_time, linestyles='dashed',
                             colors='m')

        # Formatting of the plot
        plt.xlabel('Wall clock time [s]')
        plt.ylabel('Loss')
        plt.yscale('log')
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

            # Strings used in the plot title
            ml_algorithm = this_df.iloc[0]['ML-algorithm']
            hpo_method = opt_tuple[1]
            warmstart = str(this_df.iloc[0]['warmstart'])

            # Sort DataFrame by loss values
            sorted_df = this_df.sort_values(by='losses', axis=0, ascending=True, inplace=False)
            sorted_df.reset_index(drop=True, inplace=True)

            # Find the indices of the 5 % best hyperparameter configurations
            n_best_configs = round(.05 * len(sorted_df['losses']))
            idx_best_configs = sorted_df.index[:n_best_configs]

            # New column to distinguish the 'best' and the remaining configurations
            sorted_df['Score'] = 'Rest'
            sorted_df.loc[idx_best_configs, 'Score'] = 'Best 5%'

            # Sort by descending losses to ensure that the best configurations are plotted on top
            sorted_df.sort_values(by='losses', axis=0, ascending=False, inplace=True)

            # Tuned / Optimized hyperparameters
            hyper_params = list(sorted_df['configurations'].iloc[1].keys())

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
        :return: metrics: dict, metrics: pd.DataFrame
            Dictionary that contains a dictionary with the computed metrics for each optimization tuple.
            Pandas DataFrame that contains the computed metrics.
        """

        metrics = {}
        cols = ['HPO-library', 'HPO-method', 'ML-algorithm', 'Runs', 'Evaluations', 'Workers', 'Warmstart',
                'Wall clock time [s]', 't outperform default [s]', 'Area under curve (AUC)', 'Mean(best loss)',
                'Loss ratio', 'Interquartile range(best_loss)', 't best configuration [s]',
                'Evaluations for best configuration', 'Crashes']

        metrics_df = pd.DataFrame(columns=cols)

        # Check whether a baseline has already been calculated
        if self.baseline == 0.0:
            # Compute a new baseline
            baseline_loss = self.get_baseline_loss()
            self.baseline = baseline_loss
        else:
            baseline_loss = self.baseline

        # Row index for pandas DataFrame
        idx = 1

        for opt_tuple in trial_results_dict.keys():

            this_df = trial_results_dict[opt_tuple].trial_result_df
            unique_ids = this_df['run_id'].unique()  # Unique id of each optimization run

            # Flag indicates, whether a warmstart of the HPO-method was performed successfully
            did_warmstart = trial_results_dict[opt_tuple].did_warmstart

            n_cols = len(unique_ids)
            n_rows = 0

            # Find the maximum number of function evaluations over all runs of this tuning tuple
            for uniq in unique_ids:
                num_of_evals = len(this_df.loc[this_df['run_id'] == uniq]['eval_count'])
                if num_of_evals > n_rows:
                    n_rows = num_of_evals

            # n_rows = int(len(this_df['num_of_evaluation']) / n_cols)
            best_losses = np.zeros(shape=(n_rows, n_cols))
            timestamps = np.zeros(shape=(n_rows, n_cols))

            # Count the number of algorithm crashes that occurred during optimization
            number_of_crashes_this_algo = 0

            # Iterate over all runs (with varying random seeds)
            for j in range(n_cols):
                this_subframe = this_df.loc[this_df['run_id'] == unique_ids[j]]
                this_subframe = this_subframe.sort_values(by=['eval_count'], ascending=True, inplace=False)

                # Check, whether this run was completed successfully
                if not all(this_subframe['run_successful']):
                    number_of_crashes_this_algo = number_of_crashes_this_algo + 1

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

            # Wall clock time
            wall_clock_time = max(mean_timestamps)

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
            # 5. Interquantile range of the loss of the best found configuration
            quant75 = np.nanquantile(best_losses, q=.75, axis=1)
            quant25 = np.nanquantile(best_losses, q=.25, axis=1)
            interq_range = (quant75 - quant25)[-1]

            # 6. Total number of crashes during the optimization (for each HPO-method)
            # number_of_crashes_this_algo

            # USABILITY
            if math.isnan(best_mean_loss):
                # Only crashed runs for this HPO-method
                best_idx = float('nan')
                time_best_config = float('nan')
                evals_for_best_config = float('nan')

            else:
                for eval_num in range(len(mean_trace_desc)):
                    if mean_trace_desc[eval_num] <= best_mean_loss:
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
                                           best_mean_loss=best_mean_loss,
                                           loss_ratio=loss_ratio,
                                           interquantile_range=interq_range,
                                           time_best_config=time_best_config,
                                           evals_for_best_config=evals_for_best_config,
                                           number_of_crashes=number_of_crashes_this_algo)

            # Assign the MetricsResult-object to a dictionary
            metrics[opt_tuple] = metrics_object

            # Dictionary with new metrics
            metrics_dict = {'idx': [idx],
                            'HPO-library': opt_tuple[0],
                            'HPO-method': opt_tuple[1],
                            'ML-algorithm': self.ml_algorithm,
                            'Runs': self.n_runs,
                            'Evaluations': self.n_func_evals,
                            'Workers': self.n_workers,
                            'Warmstart': did_warmstart,
                            'Wall clock time [s]': wall_clock_time,
                            't outperform default [s]': time_outperform_default,
                            'Area under curve (AUC)': auc,
                            'Mean(best loss)': best_mean_loss,
                            'Loss ratio': loss_ratio,
                            'Interquartile range(best_loss)': interq_range,
                            't best configuration [s]': time_best_config,
                            'Evaluations for best configuration': evals_for_best_config,
                            'Crashes': number_of_crashes_this_algo}

            # Create pandas DataFrame from dictionary
            this_metrics_df = pd.DataFrame.from_dict(data=metrics_dict)
            this_metrics_df.set_index(keys='idx', drop=True, inplace=True)

            # Append the new metrics / results to the whole metrics DataFrame
            metrics_df = pd.concat(objs=[metrics_df, this_metrics_df], axis=0)

            idx = idx + 1

        return metrics, metrics_df

    def get_baseline_loss(self):
        """
        Computes the loss for the default hyperparameter configuration of the ML-algorithm (baseline).
        :return:
        baseline_loss: float
            Validation loss of the baseline HP-configuration
        """
        if self.ml_algorithm == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=0)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)

        elif self.ml_algorithm == 'SVR':
            model = SVR()
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)

        elif self.ml_algorithm == 'AdaBoostRegressor':
            model = AdaBoostRegressor(random_state=0)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)

        elif self.ml_algorithm == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=0)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)

        elif self.ml_algorithm == 'LinearRegression':
            model = LinearRegression()
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)

        elif self.ml_algorithm == 'KNNRegressor':
            model = KNeighborsRegressor()
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
            model = XGBRegressor(random_state=0)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)

        else:
            raise Exception('Unknown ML-algorithm!')

        # Add remaining ML-algorithms here

        baseline_loss = self.metric(self.y_val, y_pred)

        return baseline_loss

    @staticmethod
    def train_best_model(trial_results_dict: dict):
        # TBD
        raise NotImplementedError
