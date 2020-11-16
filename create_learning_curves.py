import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


def plot_aggregated_learning_curves(logs: dict):
    # Initialize the plot figure
    fig, ax = plt.subplots()
    mean_lines = []
    max_time = 0  # necessary to limit the length of the baseline curve (default configuration)

    # Dictionary for saving the validation baseline losses of each ML algorithm
    baseline_dict = {}

    # Labels for the legend of the plot
    legend_labels = []

    # Iterate over the optimization tuples
    for opt_tuple in logs.keys():

        this_df = logs[opt_tuple]
        unique_ids = this_df['Run-ID'].unique()  # Unique id of each optimization run

        this_algo = opt_tuple[2]
        baseline_dict[this_algo] = this_df['val_baseline'][0]

        # Add legend label (ML-algorithm - HPO-technique)
        legend_labels.append(opt_tuple[2] + ' - ' + opt_tuple[3])

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
            this_subframe.reset_index(inplace=True, drop=True)

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

    ml_algorithms = [opt_tuple[2] for opt_tuple in logs.keys()]
    hpo_methods = [opt_tuple[3] for opt_tuple in logs.keys()]

    # If the log files contain HPO-results for a single ML algorithm only, add a horizontal baseline
    if len(set(ml_algorithms)) < 2:

        for i in range(len(baseline_dict.keys())):
            algo = list(baseline_dict.keys())[i]
            this_val_baseline_loss = baseline_dict[algo]
            this_val_baseline = ax.hlines(y=this_val_baseline_loss, xmin=0, xmax=max_time, linestyles='dashed',
                                          colors='deepskyblue')
            mean_lines.append(this_val_baseline)
            legend_labels.append(algo + ' - Default HPs')

    # Formatting of the plot
    plt.xlabel('Wall clock time [s]')
    plt.ylabel('Validation loss')
    plt.yscale('log')
    plt.xscale('log')

    # Add a legend
    plt.legend(mean_lines, legend_labels, loc='upper right', fontsize='small')

    # Add a title
    font = {'weight': 'semibold',
            'size': 'large'}

    title_label = 'Learning curves'
    plt.title(label=title_label, fontdict=font, loc='center')

    time_str = str(time.strftime("%Y_%m_%d %H-%M-%S", time.localtime()))

    algo_str = '_'
    for algo in set(ml_algorithms):
        algo_str = algo_str + algo

    hpo_str = '_'
    for hpo in set(hpo_methods):
        hpo_str = hpo_str + hpo

    fig_str = 'learning_curves_' + this_df['dataset'][0] + algo_str + hpo_str + '_' + time_str + '.jpg'

    return fig, fig_str


if __name__ == '__main__':

    log_path = './hpo_framework/results/temp'
    log_dict = {}

    # Iterate over all csv.files in the temp folder and load the .csv (log) files into a dictionary
    for file in os.listdir(log_path):
        if file[-4:] == '.csv':
            file_path = os.path.join(log_path, file)
            log_df = pd.read_csv(file_path, index_col=0)
            trial_id, dataset, ml_algo, hpo_method = log_df['Trial-ID'][0], log_df['dataset'][0], log_df['ML-algorithm'][0], \
                log_df['HPO-method'][0]
            log_dict[(trial_id, dataset, ml_algo, hpo_method)] = log_df

    # Plot the learning curves
    curves_fig, curves_str = plot_aggregated_learning_curves(log_dict)
    fig_path = os.path.join(log_path, curves_str)
    curves_fig.savefig(fname=fig_path)
