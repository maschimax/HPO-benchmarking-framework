import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_aggregated_learning_curves(logs: dict):
    # Initialize the plot figure
    fig, ax = plt.subplots()
    mean_lines = []
    max_time = 0  # necessary to limit the length of the baseline curve (default configuration)

    # Iterate over the optimization tuples
    for opt_tuple in logs.keys():

        this_df = logs[opt_tuple]
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

    # # Check whether a validation baseline has already been calculated
    # if self.val_baseline == 0.0:
    #     # Compute a new baseline
    #     val_baseline_loss = self.get_baseline(cv_mode=True)
    #     self.val_baseline = val_baseline_loss
    # else:
    #     val_baseline_loss = self.val_baseline

    # # Add a horizontal line for the default hyperparameter configuration of the ML-algorithm (baseline)
    # baseline = ax.hlines(val_baseline_loss, xmin=0, xmax=max_time, linestyles='dashed',
    #                      colors='m')

    # Formatting of the plot
    plt.xlabel('Wall clock time [s]')
    plt.ylabel('Validation loss')
    # plt.ylim([0.02, 1.0])
    # plt.xlim([0.05, 1000])
    plt.yscale('log')
    plt.xscale('log')
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # Add a legend
    # mean_lines.append(baseline)
    legend_labels = [this_tuple[1] for this_tuple in logs.keys()]
    # legend_labels.append('Default HPs')
    plt.legend(mean_lines, legend_labels, loc='upper right')

    # Add a title
    font = {'weight': 'semibold',
            'size': 'large'}

    # title_label = self.ml_algorithm + " - " + str(self.n_workers) + " worker(s) - " + str(self.n_runs) + " runs"
    # plt.title(label=title_label, fontdict=font, loc='center')

    return fig


if __name__ == '__main__':
    # Load log files
    log_path = './hpo_framework/results/temp'
    log_dict = {}

    for file in os.listdir(log_path):
        file_path = os.path.join(log_path, file)
        log_df = pd.read_csv(file_path, index_col=0)
        trial_id, dataset, ml_algo, hpo_method = log_df['Trial-ID'][0], log_df['dataset'][0], log_df['ML-algorithm'][0], \
            log_df['HPO-method'][0]
        log_dict[(trial_id, dataset, ml_algo, hpo_method)] = log_df

    # Use modified plot learning curves function
    curves_fig = plot_aggregated_learning_curves(log_dict)
    curves_fig.savefig(fname='test.jpg')
