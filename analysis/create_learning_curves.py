import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from matplotlib import rcParams

def plot_aggregated_learning_curves(logs: dict, show_std=False, single_mode=False):

    # IPT-colors
    colors = {'default': '#969696',  # grau
              'BOHB': '#179c7d',  # dunkelgrün
              'CMA-ES': '#ff6600',  # dunkelorange
              'RandomSearch': '#771c2d',  # bordeaux
              'Hyperband': '#438cd4',  # hellblau
              'TPE': '#005a94',  # dunkelblau
              'GPBO': '#b1c800',  # kaktus
              'SMAC': '#25b9e2',  # kaugummi
              'Fabolas': '#ffcc99',  # hellorange
              'Bohamiann': '#000000'}  # schwarz

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Arial'
    rcParams.update({'font.size': 11})

    # Initialize the plot figure
    fig, ax = plt.subplots(figsize=(6, 4))
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

        if this_algo in baseline_dict.keys():
            baseline_dict[this_algo].append(this_df['val_baseline'][0])
        else:
            baseline_dict[this_algo] = [this_df['val_baseline'][0]]

        # Add legend label (ML-algorithm - HPO-technique)
        if single_mode:
            legend_labels.append(opt_tuple[3])
        else:
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

        if show_std:

            std = np.std(best_losses, axis=1)
            upper = mean_trace_desc + std
            lower = mean_trace_desc - std

        else:

            # 25% and 75% loss quantile for each point (function evaluation)
            upper = np.nanquantile(best_losses, q=.25, axis=1)
            lower = np.nanquantile(best_losses, q=.75, axis=1)

        # Compute average timestamps
        mean_timestamps = np.nanmean(timestamps, axis=1)

        if max(mean_timestamps) > max_time:
            max_time = max(mean_timestamps)

        # Plot the mean loss over time
        mean_line = ax.plot(mean_timestamps, mean_trace_desc, c=colors[opt_tuple[3]])
        mean_lines.append(mean_line[0])

        # Colored area to visualize the inter-quantile area
        ax.fill_between(x=mean_timestamps, y1=upper,
                        y2=lower, color=colors[opt_tuple[3]], alpha=0.2)

    ml_algorithms = [opt_tuple[2] for opt_tuple in logs.keys()]
    hpo_methods = [opt_tuple[3] for opt_tuple in logs.keys()]

    # If the log files contain HPO-results for a single ML algorithm only, add a horizontal baseline
    if len(set(ml_algorithms)) < 2:

        for i in range(len(baseline_dict.keys())):
            algo = list(baseline_dict.keys())[i]
            this_val_baseline_loss = np.nanmean(baseline_dict[algo])
            this_val_baseline = ax.hlines(y=this_val_baseline_loss, xmin=0, xmax=max_time, linestyles='dashed',
                                          colors=colors['default'])
            mean_lines.append(this_val_baseline)

            if single_mode:
                legend_labels.append('Default HPs')
            else:
                legend_labels.append(algo + ' - Default HPs')

    # Formatting of the plot
    ax.set_xlabel('Wall clock time [s]', fontweight='semibold', fontsize=11, fontname='Arial')
    ax.set_ylabel('Validation loss', fontweight='semibold', fontsize=11, fontname='Arial')
    plt.yscale('log')
    plt.xscale('log')

    # Add a legend
    plt.legend(mean_lines, legend_labels, loc='upper right', fontsize='small')

    # # Add a title
    # font = {'weight': 'semibold',
    #         'size': 11}
    #
    # title_label = 'Learning curves'
    # plt.title(label=title_label, fontdict=font, loc='center')

    # time_str = str(time.strftime("%Y_%m_%d", time.localtime()))

    algo_str = '_'
    for algo in set(ml_algorithms):
        algo_str = algo_str + algo

    hpo_str = '_'
    for hpo in set(hpo_methods):
        hpo_str = hpo_str + hpo

    fig_str_jpg = 'learning_curves_' + this_df['dataset'][0] + algo_str + hpo_str + '.jpg'
    fig_str_svg = 'learning_curves_' + this_df['dataset'][0] + algo_str + hpo_str + '.svg'

    return fig, fig_str_jpg, fig_str_svg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script for creating learning curves from log (.csv) files")

    parser.add_argument('--std', type=str, help='Plot mean +/- standard deviation', default='No',
                        choices=['Yes', 'No'])

    parser.add_argument('--single_algo_mode', type=str, help='Single ML algorithm (Yes/No)', default='No',
                        choices=['Yes', 'No'])

    args = parser.parse_args()

    if args.std == 'Yes':
        show_std = True
    else:
        show_std = False

    if args.single_algo_mode == 'Yes':
        single_mode = True
    else:
        single_mode = False

    os.chdir('..')
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
    curves_fig, curves_str_jpg, curves_str_svg = plot_aggregated_learning_curves(log_dict, show_std=show_std,
                                                                                 single_mode=single_mode)
    fig_path_jpg = os.path.join(log_path, curves_str_jpg)
    fig_path_svg = os.path.join(log_path, curves_str_svg)
    curves_fig.savefig(fname=fig_path_jpg, bbox_inches='tight')
    curves_fig.savefig(fname=fig_path_svg, bbox_inches='tight')
