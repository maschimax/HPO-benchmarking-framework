import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

dataset = 'turbofan'

metrics_path = './hpo_framework/results/' + dataset + '/metrics_with_cuts_' + dataset + '.csv'
metrics_df = pd.read_csv(metrics_path, index_col=0)

hpo_techniques = metrics_df['HPO-method'].unique()
ml_algorithms = metrics_df['ML-algorithm'].unique()

# Plot params
bar_width = 0.6
font_name = 'Arial'
font_size = 10
large_fig_size = (7, 4)
small_fig_size = (3, 2)
xtick_rotation = 90

########################################################################################################################
# Summary table for each setup variant
setup_variants = [(1, False), (8, False), (1, True)]

# Flag indicating, whether the Final Performance should be assessed based on a time budget restriction or a
# restriction of the number of function evaluations
time_budget_restriction = True

analysis_dict = {'Max Cut Loss': {'1st metric': 'Max Cut Test Loss',
                                  '2nd metric': 'Max Cut Validation Loss',
                                  'Budget col': 'Max Cut Time Budget [s]'},
                 '2nd Cut Loss': {'1st metric': '2nd Cut Validation Loss',
                                  'Budget col': '2nd Cut Time Budget [s]'},
                 '3rd Cut Loss': {'1st metric': '3rd Cut Validation Loss',
                                  'Budget col': '3rd Cut Time Budget [s]'},
                 '4th Cut Loss': {'1st metric': '4th Cut Validation Loss',
                                  'Budget col': '4th Cut Time Budget [s]'},
                 'Max Cut AUC': {'1st metric': 'Max Cut AUC',
                                 'Budget col': 'Max Cut Time Budget [s]'},
                 'Outperform Baseline': {'1st metric': 't outperform default [s]',
                                         'Budget col': 'Max Cut Time Budget [s]'}}

algo_list = []
worker_list = []
wst_list = []
rank_list = []
hpo_list = []
first_metric_name = []
first_metric_list = []
scaled_list = []
second_metric_name = []
second_metric_list = []
budget_list = []

avg_time_per_eval_list = []
dim_list = []
cpl_class_list = []

# Iterate over setup variants
for this_setup in setup_variants:

    for this_analysis in analysis_dict.keys():

        setup_df = metrics_df.loc[(metrics_df['Workers'] == this_setup[0]) &
                                  (metrics_df['Warmstart'] == this_setup[1]), :]

        ml_algos = setup_df['ML-algorithm'].unique()

        for this_algo in ml_algos:

            # Filter for ML-algorithm -> HPO Use Case
            use_case_df = setup_df.loc[setup_df['ML-algorithm'] == this_algo, :]

            # Continue, if there is no data for this ML algorithm <-> analysis combination
            if use_case_df[analysis_dict[this_analysis]['1st metric']].isnull().all():
                continue

            # Correct negative values of 't outperform default [s]' -> may occur on the warm started setup
            if this_setup[1]:
                for idx, row in use_case_df.iterrows():
                    if row['t outperform default [s]'] < 0.0:
                        use_case_df.loc[idx, 't outperform default [s]'] = 0.0

            # Add row for HPO technique 'Default HPs'
            default_test_loss = np.nanmean(use_case_df.loc[:, 'Test baseline'])
            default_val_loss = np.nanmean(use_case_df.loc[:, 'Validation baseline'])
            default_auc = np.nanmean(use_case_df.loc[:, 'Validation baseline'])
            default_t_outperform = np.inf

            default_row = {'HPO-method': 'Default HPs',
                           'ML-algorithm': this_algo,
                           'Workers': this_setup[0],
                           'Warmstart': this_setup[1],
                           'Mean (final validation loss)': default_val_loss,
                           'Mean (final test loss)': default_test_loss,
                           'Area under curve (AUC)': default_auc,
                           't outperform default [s]': default_t_outperform}

            this_metric = analysis_dict[this_analysis]['1st metric']

            if this_metric == 'Max Cut Test Loss':

                default_row[analysis_dict[this_analysis]['1st metric']] = default_test_loss
                default_row[analysis_dict[this_analysis]['2nd metric']] = default_val_loss

            elif this_metric in ['2nd Cut Validation Loss', '3rd Cut Validation Loss', '4th Cut Validation Loss']:

                default_row[analysis_dict[this_analysis]['1st metric']] = default_val_loss

            elif this_metric == 'Max Cut AUC':

                default_row[analysis_dict[this_analysis]['1st metric']] = default_auc

            elif this_metric == 't outperform default [s]':

                default_row[analysis_dict[this_analysis]['1st metric']] = default_t_outperform

            else:

                raise Exception('Unknown analysis type for default HPs!')

            default_df = pd.DataFrame(default_row, index=[len(use_case_df)])
            use_case_df = pd.concat(objs=[use_case_df, default_df], axis=0, ignore_index=True)

            analysis_df = use_case_df.sort_values(by=analysis_dict[this_analysis]['1st metric'], axis=0, inplace=False,
                                                  ascending=True, na_position='last')

            algo_list += ([this_algo] * len(analysis_df['HPO-method']))
            worker_list += ([this_setup[0]] * len(analysis_df['HPO-method']))
            wst_list += ([this_setup[1]] * len(analysis_df['HPO-method']))
            rank_list += (list(range(1, len(analysis_df['HPO-method']) + 1)))
            hpo_list += (list(analysis_df['HPO-method']))
            first_metric_name += ([analysis_dict[this_analysis]['1st metric']] * len(analysis_df['HPO-method']))
            first_metric_list += (list(analysis_df[analysis_dict[this_analysis]['1st metric']]))
            budget_list += (list(analysis_df[analysis_dict[this_analysis]['Budget col']]))
            if '2nd metric' in analysis_dict[this_analysis].keys():
                second_metric_name += ([analysis_dict[this_analysis]['2nd metric']] * len(analysis_df['HPO-method']))
                second_metric_list += (list(analysis_df[analysis_dict[this_analysis]['2nd metric']]))
            else:
                second_metric_name += ([np.nan] * len(analysis_df['HPO-method']))
                second_metric_list += ([np.nan] * len(analysis_df['HPO-method']))

            # Compute the scaled deviation for the 1st metric
            metric_arr = analysis_df[analysis_dict[this_analysis]['1st metric']].to_numpy()
            min_value = np.nanmin(metric_arr)
            max_value = np.nanmax(metric_arr[metric_arr != np.inf])

            if max_value - min_value > 0.0:

                scaled_deviation = [(this_value - min_value) / (max_value - min_value) for this_value
                                    in list(metric_arr)]

            # Avoid division by zero
            else:

                scaled_deviation = [np.inf] * len(metric_arr)

            scaled_list += scaled_deviation

            # Average time per function evaluation of Random Search (on this setup variant)
            wc_time_rs = analysis_df.loc[analysis_df['HPO-method'] == 'RandomSearch',
                                         'Wall clock time [s]'].to_numpy()[0]

            n_evals = analysis_df.loc[analysis_df['HPO-method'] == 'RandomSearch', 'Evaluations'].to_numpy()[0]
            time_per_eval = round(wc_time_rs / n_evals, 2)
            avg_time_per_eval_list += ([time_per_eval] * len(analysis_df['HPO-method']))

            # Number of each HP type
            num_cont_hps = np.nanmean(analysis_df.loc[:, '# cont. HPs'].to_numpy())
            num_int_hps = np.nanmean(analysis_df.loc[:, '# int. HPs'].to_numpy())
            num_cat_hps = np.nanmean(analysis_df.loc[:, '# cat. HPs'].to_numpy())

            # Dimensionality of the configuration space for this ML algorithm
            dim_list += ([num_cont_hps + num_int_hps + num_cat_hps] * len(analysis_df['HPO-method']))

            # Check complexity class
            if num_cat_hps > 0:
                cplx_class = 3
            elif num_int_hps > 0:
                cplx_class = 2
            else:
                cplx_class = 1

            # Append the complexity class of this ML algorithm
            cpl_class_list += ([cplx_class] * len(analysis_df['HPO-method']))

            # TODO: Parallelized Setup -> effective use of parallel resources
            # # For parallelized setup -> compute speed up factor due to parallelization
            # if this_setup[0] == 8:
            #
            #     # Filter for single worker, no warm start setup
            #     single_sub_frame = metrics_df.loc[
            #                        (metrics_df['ML-algorithm'] == this_algo) & (metrics_df['Workers'] == 1)
            #                        & (metrics_df['Warmstart'] == False), :]
            #
            #     # Filter for 8 parallel workers, no warm start setup
            #     para_sub_frame = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) & (metrics_df['Workers'] == 8)
            #                                     & (metrics_df['Warmstart'] == False), :]
            #
            #     # Iterate over HPO-techniques
            #     for this_tech in para_sub_frame['HPO-method'].unique():
            #         # Wall clock time on single worker setup
            #         single_wc_time = single_sub_frame.loc[single_sub_frame['HPO-method'] == this_tech,
            #                                               'Wall clock time [s]'].values[0]
            #
            #         # Wall clock time on setup with 8 parallel workers
            #         para_wc_time = para_sub_frame.loc[para_sub_frame['HPO-method'] == this_tech,
            #                                           'Wall clock time [s]'].values[0]
            #
            #         # Speed up factor
            #         this_speed_up = round(single_wc_time / para_wc_time, 2)
            #
            #         para_sub_frame.loc[para_sub_frame['HPO-method'] == this_tech, 'Speed up factor'] = this_speed_up
            #
            #     # Default baselines on single worker setup
            #     single_validation_baseline = np.nanmean(single_sub_frame.loc[:, 'Validation baseline'].to_numpy())
            #     single_test_baseline = np.nanmean(single_sub_frame.loc[:, 'Test baseline'].to_numpy())
            #
            #     # Add row for Default HPs
            #     para_default_row = {'HPO-method': 'Default',
            #                         'ML-algorithm': this_algo,
            #                         'Workers': this_setup[0],
            #                         'Warmstart': this_setup[1],
            #                         'Mean (final validation loss)': single_validation_baseline,
            #                         'Mean (final test loss)': single_test_baseline,
            #                         'Validation baseline': single_validation_baseline,
            #                         'Test baseline': single_test_baseline,
            #                         'Min.avg.test loss in time budget': single_test_baseline,
            #                         'Speed up factor': 1.0}
            #
            #     para_default_df = pd.DataFrame(para_default_row, index=[len(para_sub_frame)])
            #
            #     # Append new row to DataFrame
            #     para_sub_frame = pd.concat(objs=[para_sub_frame, para_default_df], axis=0, ignore_index=True)
            #
            #     para_sub_frame.sort_values(by='Speed up factor', axis=0, inplace=True, ascending=False,
            #                                na_position='last')
            #
            #     para_list_rank += (list(range(1, len(para_sub_frame['HPO-method']) + 1)))
            #     para_list_tech += (list(para_sub_frame['HPO-method']))
            #     para_list_val += (list(para_sub_frame['Speed up factor']))
            #     para_wct_list += (list(para_sub_frame['Wall clock time [s]']))
            #
            #     # Compute the deviation from the maximum speed up scaled between 0 and 1
            #     speed_up_arr = para_sub_frame['Speed up factor'].to_numpy()
            #     max_speed_up = np.nanmax(speed_up_arr[speed_up_arr != np.inf])
            #     min_speed_up = np.nanmin(speed_up_arr)
            #
            #     para_list_rel += [(max_speed_up - this_speed_up) / (max_speed_up - min_speed_up) for this_speed_up in
            #                       list(para_sub_frame['Speed up factor'])]

# Create pd.DataFrame and save to .csv file
ranked_dict = {'ML-algorithm': algo_list,
               'Workers': worker_list,
               'Warm start': wst_list,
               '1st metric': first_metric_name,
               '1st metric rank': rank_list,
               '1st metric HPO-technique': hpo_list,
               '1st metric value': first_metric_list,
               '1st metric scaled deviation': scaled_list,
               'Time budget [s]': budget_list,
               'Time per eval on this setup (RS)': avg_time_per_eval_list,
               'Number of HPs': dim_list,
               'HP complexity': cpl_class_list,
               '2nd metric': second_metric_name,
               '2nd metric value': second_metric_list}

analysis_ranked_df = pd.DataFrame(ranked_dict)

save_dir = './hpo_framework/results/' + dataset + '/RankingAnalysis'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

file_name = dataset + '_ranked_metrics.csv'

analysis_ranked_df.to_csv(os.path.join(save_dir, file_name))

exit(0)

########################################################################################################################
# 1. Effective use of parallel resources
hpo_list = []
ml_list = []
speed_up_list = []
time_per_eval_list = []

# Iterate over ML algorithms
for this_algo in ml_algorithms:

    # TODO: How to deal with Keras (no parallel setup)
    if this_algo == 'KerasRegressor' or this_algo == 'KerasClassifier':
        continue

    # Iterate over HPO techniques
    for this_tech in hpo_techniques:

        # TODO: Add 8 worker metrics for Fabolas and Bohamiann
        if this_tech == 'Fabolas' or this_tech == 'Bohamiann':
            continue

        # Filter for ML algorithm, HPO technique and no warm start
        sub_frame = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) & (metrics_df['HPO-method'] == this_tech)
                                   & (metrics_df['Warmstart'] == False), :]

        # Wall clock times for single and multiple worker setup
        single_worker_time = sub_frame.loc[sub_frame['Workers'] == 1, 'Wall clock time [s]'].values
        multiple_worker_time = sub_frame.loc[sub_frame['Workers'] == 8, 'Wall clock time [s]'].values

        # Calculate the average time per evaluation for the single worker setup
        single_worker_avg_time_per_eval = float(sub_frame.loc[sub_frame['Workers'] == 1, 'Wall clock time [s]'].values /
                                                sub_frame.loc[sub_frame['Workers'] == 1, 'Evaluations'].values)

        # Calculate the speed up
        speed_up_factor = float(single_worker_time / multiple_worker_time)

        # Append the results
        hpo_list.append(this_tech)
        ml_list.append(this_algo)
        speed_up_list.append(speed_up_factor)
        time_per_eval_list.append(single_worker_avg_time_per_eval)

# Create a pandas DataFrame to store the results
speed_up_df = pd.DataFrame({'HPO-method': hpo_list,
                            'ML-algorithm': ml_list,
                            'Speed-up': speed_up_list,
                            'Avg. time per evaluation [s]': time_per_eval_list})

# Compute the average speed up, that each HPO technique achieves due to parallelization
avg_speed_up_dict = {}
for this_tech in speed_up_df['HPO-method'].unique():
    this_avg = np.nanmean(speed_up_df.loc[speed_up_df['HPO-method'] == this_tech, 'Speed-up'].values)
    avg_speed_up_dict[this_tech] = this_avg

# Boxplots to visualize the distribution of speed up factors for each HPO technique
box_labels = []
box_vec = []
for this_tech in speed_up_df['HPO-method'].unique():
    box_labels.append(this_tech)
    speed_up_factors = speed_up_df.loc[speed_up_df['HPO-method'] == this_tech, 'Speed-up'].values
    speed_up_factors = speed_up_factors[~np.isnan(speed_up_factors)]
    box_vec.append(list(speed_up_factors))

box_fig, box_ax = plt.subplots(figsize=large_fig_size)

box_ax.boxplot(x=box_vec, labels=box_labels, showfliers=True)
box_ax.hlines(y=1.0, xmin=1, xmax=len(box_labels), linestyles='dashed', colors='deepskyblue')

box_ax.set_ylabel('Speed up factor', fontweight='semibold', fontsize=font_size,
                  color='#969696', fontname=font_name)

box_ax.spines['bottom'].set_color('#969696')
box_ax.spines['top'].set_color('#969696')
box_ax.spines['right'].set_color('#969696')
box_ax.spines['left'].set_color('#969696')
box_ax.tick_params(axis='x', colors='#969696', rotation=90)
box_ax.tick_params(axis='y', colors='#969696')

if not os.path.isdir('./hpo_framework/results/' + dataset + '/Parallelization/'):
    os.mkdir('./hpo_framework/results/' + dataset + '/Parallelization/')

fig_name1 = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_speed_up_box' + '.jpg'
fig_name2 = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_speed_up_box' + '.svg'
box_fig.savefig(fig_name1, bbox_inches='tight')
box_fig.savefig(fig_name2, bbox_inches='tight')

# Sort the dictionary according to the speed up achieved (descending)
sorted_avg_speed_up_dict = dict(sorted(avg_speed_up_dict.items(), key=lambda item: item[1], reverse=True))

# Bar plot to visualize the results
fig, ax = plt.subplots(figsize=large_fig_size)
plt.bar(x=sorted_avg_speed_up_dict.keys(), height=sorted_avg_speed_up_dict.values(), color='#179c7d', width=bar_width)

# Formatting
ax.set_xlabel('HPO technique', fontweight='semibold', fontsize=font_size, color='#969696', fontname=font_name)
ax.set_ylabel('Average speed up factor', fontweight='semibold', fontsize=font_size, color='#969696', fontname=font_name)

ax.spines['bottom'].set_color('#969696')
ax.spines['top'].set_color('#969696')
ax.spines['right'].set_color('#969696')
ax.spines['left'].set_color('#969696')
ax.tick_params(axis='x', colors='#969696', rotation=xtick_rotation)
ax.tick_params(axis='y', colors='#969696')

# Display the values in each bar
for idx, val in enumerate(sorted_avg_speed_up_dict.values()):
    ax.text(list(sorted_avg_speed_up_dict.keys())[idx], 0.1, round(float(val), 2), color='white', ha='center',
            fontname=font_name, fontsize=font_size - 1)

if not os.path.isdir('./hpo_framework/results/' + dataset + '/Parallelization/'):
    os.mkdir('./hpo_framework/results/' + dataset + '/Parallelization/')

fig_name1 = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_parallelization.jpg'
fig_name2 = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_parallelization.svg'
plt.savefig(fig_name1, bbox_inches='tight')
plt.savefig(fig_name2, bbox_inches='tight')

# Create new column, that stores the maximum speed up achieved for each HPO technique
for this_tech in speed_up_df['HPO-method'].unique():
    # Index of row, where this_tech achieved the maximum speed up
    max_idx = speed_up_df.loc[speed_up_df['HPO-method'] == this_tech, 'Speed-up'].idxmax(axis=0, skipna=True)

    # Maximum speed up factor of this_tech
    max_speed_up = speed_up_df.loc[speed_up_df['HPO-method'] == this_tech, 'Speed-up'].max(axis=0, skipna=True)

    # ML algorithm, where this_tech achieved the maximum speed up
    max_algo = speed_up_df.loc[max_idx, 'ML-algorithm']

    speed_up_df.loc[speed_up_df['HPO-method'] == this_tech, 'Max speed-up ML algorithm per HPO technique'] = max_algo
    speed_up_df.loc[speed_up_df['HPO-method'] == this_tech, 'Max speed-up per HPO technique'] = max_speed_up

# Save the table in .csv format
table_name = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_speed_up_table.csv'
speed_up_df.to_csv(table_name)

# Scatter plot to visualize how the speed up depends on the avg. time per evaluation
fig, ax = plt.subplots(figsize=large_fig_size)

# Focus on optuna methods
tpe_df = speed_up_df.loc[(speed_up_df['HPO-method'] == 'TPE'), :]
rs_df = speed_up_df.loc[(speed_up_df['HPO-method'] == 'RandomSearch'), :]
cma_df = speed_up_df.loc[(speed_up_df['HPO-method'] == 'CMA-ES'), :]

# Compute Pearon correlation coefficient
tpe_corr = tpe_df.loc[:, ['Avg. time per evaluation [s]', 'Speed-up']].corr(method='pearson')
rs_corr = rs_df.loc[:, ['Avg. time per evaluation [s]', 'Speed-up']].corr(method='pearson')
cma_corr = cma_df.loc[:, ['Avg. time per evaluation [s]', 'Speed-up']].corr(method='pearson')

tpe_scatter = ax.scatter(x=tpe_df['Avg. time per evaluation [s]'], y=tpe_df['Speed-up'], c='#179c7d')
rs_scatter = ax.scatter(x=rs_df['Avg. time per evaluation [s]'], y=rs_df['Speed-up'], c='#ff6600')
cma_scatter = ax.scatter(x=cma_df['Avg. time per evaluation [s]'], y=cma_df['Speed-up'], c='#0062a5')

max_time = max(tpe_df['Avg. time per evaluation [s]'].max(), rs_df['Avg. time per evaluation [s]'].max(),
               cma_df['Avg. time per evaluation [s]'].max())

ax.hlines(y=1.0, xmin=0.0, xmax=max_time, linestyles='dashed', colors='deepskyblue')

# Formatting
ax.set_xlabel('Average time per function evaluation [s]', fontweight='semibold', fontsize=font_size, color='#969696',
              fontname=font_name)
ax.set_ylabel('Speed up factor', fontweight='semibold', fontsize=font_size, color='#969696', fontname=font_name)

ax.spines['bottom'].set_color('#969696')
ax.spines['top'].set_color('#969696')
ax.spines['right'].set_color('#969696')
ax.spines['left'].set_color('#969696')
ax.tick_params(axis='x', colors='#969696')
ax.tick_params(axis='y', colors='#969696')

ax.legend((tpe_scatter, rs_scatter, cma_scatter), ('TPE', 'RandomSearch', 'CMA-ES'))
plt.xscale('log')

fig_name1 = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_speed_vs_avg_time.jpg'
fig_name2 = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_speed_vs_avg_time.svg'
plt.savefig(fig_name1, bbox_inches='tight')
plt.savefig(fig_name2, bbox_inches='tight')

########################################################################################################################
# 2. Robustness

hpo_list = []
run_list = []
crash_list = []
crash_percentage_lst = []

# Iterate over HPO techniques
for this_tech in hpo_techniques:
    # Filter for HPO technique
    sub_frame = metrics_df.loc[(metrics_df['HPO-method'] == this_tech), :]

    n_runs = sub_frame['Runs'].sum()
    n_crashes = sub_frame['Crashes'].sum()
    crash_percentage = round(100 * n_crashes / n_runs, 2)

    hpo_list.append(this_tech)
    run_list.append(n_runs)
    crash_list.append(n_crashes)
    crash_percentage_lst.append(crash_percentage)

# Create a pandas DataFrame to store the results
crash_df = pd.DataFrame({'HPO-method': hpo_list,
                         'Runs': run_list,
                         'Crashed': crash_list,
                         'Crashes [%]': crash_percentage_lst})

crash_dict = {}
for this_tech in crash_df['HPO-method'].unique():
    crash_dict[this_tech] = crash_df.loc[crash_df['HPO-method'] == this_tech, 'Crashes [%]'].values[0]

sorted_crash_dict = dict(sorted(crash_dict.items(), key=lambda item: item[1], reverse=True))

# # Remove HPO techniques without crashes
# non_empty_crash_dict = sorted_crash_dict.copy()
# for this_tech in sorted_crash_dict.keys():
#     if sorted_crash_dict[this_tech] == 0.0:
#         del non_empty_crash_dict[this_tech]

fig, ax = plt.subplots(figsize=(7, 4))
plt.bar(x=sorted_crash_dict.keys(), height=sorted_crash_dict.values(), color='#179c7d', width=bar_width)

# Formatting
ax.set_xlabel('HPO technique', fontweight='semibold', fontsize=font_size, color='#969696', fontname=font_name)
ax.set_ylabel('Share of crashed runs [%]', fontweight='semibold', fontsize=font_size, color='#969696',
              fontname=font_name)

ax.spines['bottom'].set_color('#969696')
ax.spines['top'].set_color('#969696')
ax.spines['right'].set_color('#969696')
ax.spines['left'].set_color('#969696')
ax.tick_params(axis='x', colors='#969696', rotation=xtick_rotation)
ax.tick_params(axis='y', colors='#969696')

# Display the values in each bar
for idx, val in enumerate(sorted_crash_dict.values()):
    ax.text(list(sorted_crash_dict.keys())[idx], .8, round(float(val), 2), color='white', ha='center',
            fontname=font_name, fontsize=font_size - 1)

if not os.path.isdir('./hpo_framework/results/' + dataset + '/Robustness/'):
    os.mkdir('./hpo_framework/results/' + dataset + '/Robustness/')

fig_name1 = './hpo_framework/results/' + dataset + '/Robustness/' + dataset + '_robustness.jpg'
fig_name2 = './hpo_framework/results/' + dataset + '/Robustness/' + dataset + '_robustness.svg'
plt.savefig(fig_name1, bbox_inches='tight')
plt.savefig(fig_name2, bbox_inches='tight')

# TODO: Save results in table format

########################################################################################################################
# 3. Final Performance

# Setup tuples (n_workers, warm_start(Yes/No)
setups = [(1, False), (8, False), (1, True)]

# Iterate over setup variants
for setup_tuple in setups:

    # Setup for this iteration
    n_workers = setup_tuple[0]
    do_warm_start = setup_tuple[1]

    hpo_list = []
    ml_list = []
    rank_list = []

    # Iterate over ML algorithms
    for this_algo in ml_algorithms:

        # Filter according to the ML algorithm and the setup variables
        sub_frame = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) & (metrics_df['Workers'] == n_workers) &
                                   (metrics_df['Warmstart'] == do_warm_start), :]

        # Rank the HPO techniques according to the mean (final test loss)
        sub_frame['final_rank'] = sub_frame.loc[:, 'Mean (final test loss)'].rank(axis=0, method='average',
                                                                                  na_option='bottom',
                                                                                  ascending=True)

        for this_tech in sub_frame['HPO-method'].unique():
            this_rank = sub_frame.loc[(sub_frame['HPO-method'] == this_tech), 'final_rank'].values[0]

            hpo_list.append(this_tech)
            ml_list.append(this_algo)
            rank_list.append(this_rank)

    # Create a pandas DataFrame to store the results
    rank_df = pd.DataFrame({'HPO-method': hpo_list,
                            'ML-algorithm': ml_list,
                            'Final Performance Rank': rank_list})

    # Compute the average rank for each HPO technique
    rank_dict = {}
    for this_tech in rank_df['HPO-method'].unique():
        this_rank = np.nanmean(rank_df.loc[rank_df['HPO-method'] == this_tech, 'Final Performance Rank'].values)
        rank_dict[this_tech] = this_rank

    # Sort the dictionary according to the rank achieved (descending)
    sorted_rank_dict = dict(sorted(rank_dict.items(), key=lambda item: item[1], reverse=False))

    # Bar plot to visualize the results
    fig, ax = plt.subplots(figsize=small_fig_size)
    plt.bar(x=sorted_rank_dict.keys(), height=sorted_rank_dict.values(), color='#179c7d', width=bar_width)

    # Formatting
    # ax.set_xlabel('HPO technique', fontweight='semibold', fontsize=font_size -1, color='#969696', fontname=font_name)
    ax.set_ylabel('Average Rank', fontweight='semibold', fontsize=font_size - 1, color='#969696',
                  fontname=font_name)
    ax.spines['bottom'].set_color('#969696')
    ax.spines['top'].set_color('#969696')
    ax.spines['right'].set_color('#969696')
    ax.spines['left'].set_color('#969696')
    ax.tick_params(axis='x', colors='#969696', labelsize=font_size - 1, rotation=xtick_rotation)
    ax.tick_params(axis='y', colors='#969696', labelsize=font_size - 1)

    # Display the values in each bar
    for idx, val in enumerate(sorted_rank_dict.values()):
        ax.text(list(sorted_rank_dict.keys())[idx], 0.25, round(float(val), 2), color='white', ha='center', va='bottom',
                fontname=font_name, fontsize=font_size - 1, rotation=90)

    if do_warm_start:
        warm_str = 'WarmStart'
    else:
        warm_str = 'NoWarmStart'

    fig_str = '_FinalPerformance_' + str(n_workers) + 'Workers_' + warm_str

    if not os.path.isdir('./hpo_framework/results/' + dataset + '/FinalPerformance/'):
        os.mkdir('./hpo_framework/results/' + dataset + '/FinalPerformance/')

    fig_name1 = './hpo_framework/results/' + dataset + '/FinalPerformance/' + dataset + fig_str + '.jpg'
    fig_name2 = './hpo_framework/results/' + dataset + '/FinalPerformance/' + dataset + fig_str + '.svg'
    plt.savefig(fig_name1, bbox_inches='tight')
    plt.savefig(fig_name2, bbox_inches='tight')

# Final Performance Table
hpo_list = []
loss_list = []
algo_list = []
worker_list = []
warm_start_list = []

for this_tech in metrics_df['HPO-method'].unique():
    best_idx = metrics_df.loc[metrics_df['HPO-method'] == this_tech, 'Mean (final test loss)'].idxmin(axis=0,
                                                                                                      skipna=True)
    best_loss = metrics_df.loc[best_idx, 'Mean (final test loss)']
    best_algo = metrics_df.loc[best_idx, 'ML-algorithm']
    n_workers = metrics_df.loc[best_idx, 'Workers']
    do_warm_start = metrics_df.loc[best_idx, 'Warmstart']

    hpo_list.append(this_tech)
    loss_list.append(best_loss)
    algo_list.append(best_algo)
    worker_list.append(n_workers)
    warm_start_list.append(do_warm_start)

best_loss_df = pd.DataFrame({'HPO-method': hpo_list,
                             'ML-algorithm': algo_list,
                             'Best Test Loss:': loss_list,
                             'Workers': worker_list,
                             'Warmstart': warm_start_list})

# Save the table in .csv format
table_name = './hpo_framework/results/' + dataset + '/FinalPerformance/' + dataset + '_best_loss_table.csv'
best_loss_df.to_csv(table_name)

# TODO: Save results in table format

########################################################################################################################
# 4. Anytime Performance

# Setup tuples (n_workers, warm_start(Yes/No)
setups = [(1, False), (8, False), (1, True)]

# Iterate over setup variants
for setup_tuple in setups:

    # Setup for this iteration
    n_workers = setup_tuple[0]
    do_warm_start = setup_tuple[1]

    hpo_list = []
    ml_list = []
    rank_list = []

    # Iterate over ML algorithms
    for this_algo in ml_algorithms:

        # Filter according to the ML algorithm and the setup variables
        sub_frame = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) & (metrics_df['Workers'] == n_workers) &
                                   (metrics_df['Warmstart'] == do_warm_start), :]

        # Rank the HPO techniques according to the wall-clock-time, that was necessary to outperform the baseline
        sub_frame['anytime_rank'] = sub_frame.loc[:, 't outperform default [s]'].rank(axis=0, method='average',
                                                                                      na_option='bottom',
                                                                                      ascending=True)

        for this_tech in sub_frame['HPO-method'].unique():
            this_rank = sub_frame.loc[(sub_frame['HPO-method'] == this_tech), 'anytime_rank'].values[0]

            hpo_list.append(this_tech)
            ml_list.append(this_algo)
            rank_list.append(this_rank)

    # Create a pandas DataFrame to store the results
    rank_df = pd.DataFrame({'HPO-method': hpo_list,
                            'ML-algorithm': ml_list,
                            'Anytime Performance Rank': rank_list})

    # Compute the average rank for each HPO technique
    rank_dict = {}
    for this_tech in rank_df['HPO-method'].unique():
        this_rank = np.nanmean(rank_df.loc[rank_df['HPO-method'] == this_tech, 'Anytime Performance Rank'].values)
        rank_dict[this_tech] = this_rank

    # Sort the dictionary according to the rank achieved (descending)
    sorted_rank_dict = dict(sorted(rank_dict.items(), key=lambda item: item[1], reverse=False))

    # Bar plot to visualize the results
    fig, ax = plt.subplots(figsize=small_fig_size)
    plt.bar(x=sorted_rank_dict.keys(), height=sorted_rank_dict.values(), color='#179c7d', width=bar_width)

    # Formatting
    # ax.set_xlabel('HPO technique', fontweight='semibold', fontsize=font_size - 1, color='#969696',
    #               fontname=font_name)
    ax.set_ylabel('Average Rank', fontweight='semibold', fontsize=font_size - 1, color='#969696',
                  fontname=font_name)
    ax.spines['bottom'].set_color('#969696')
    ax.spines['top'].set_color('#969696')
    ax.spines['right'].set_color('#969696')
    ax.spines['left'].set_color('#969696')
    ax.tick_params(axis='x', colors='#969696', rotation=xtick_rotation, labelsize=font_size - 1)
    ax.tick_params(axis='y', colors='#969696', labelsize=font_size - 1)

    # Display the values in each bar
    for idx, val in enumerate(sorted_rank_dict.values()):
        ax.text(list(sorted_rank_dict.keys())[idx], 0.25, round(float(val), 2), color='white', ha='center', va='bottom',
                fontsize=font_size - 1, fontname=font_name, rotation=90)

    if do_warm_start:
        warm_str = 'WarmStart'
    else:
        warm_str = 'NoWarmStart'

    fig_str = '_AnytimePerformance_' + str(n_workers) + 'Workers_' + warm_str

    if not os.path.isdir('./hpo_framework/results/' + dataset + '/AnytimePerformance/'):
        os.mkdir('./hpo_framework/results/' + dataset + '/AnytimePerformance/')

    fig_name1 = './hpo_framework/results/' + dataset + '/AnytimePerformance/' + dataset + fig_str + '.jpg'
    fig_name2 = './hpo_framework/results/' + dataset + '/AnytimePerformance/' + dataset + fig_str + '.svg'
    plt.savefig(fig_name1, bbox_inches='tight')
    plt.savefig(fig_name2, bbox_inches='tight')

    # TODO: Save results in table format

########################################################################################################################
# 5. Scalability

dim_threshold = 4  # num HPs > 4 -> high dimensional

dim_classes = ['LowDimensional', 'HighDimensional']

scale_metrics_df = metrics_df.copy(deep=True)

scale_metrics_df['num_params'] = scale_metrics_df.loc[:, ['# cont. HPs', '# int. HPs', '# cat. HPs']].sum(axis=1)

# Iterate over the dimensionality classes
for dim_class in dim_classes:

    hpo_list = []
    ml_list = []
    final_rank_list = []
    anytime_rank_list = []

    # Iterate over ML algorithms
    for this_algo in ml_algorithms:

        # Low dimensional HP space (dimensionality <= dim_threshold)
        if dim_class == 'LowDimensional':

            # Filter according to the ML algorithm, the setup variables and the dimensionality
            sub_frame = scale_metrics_df.loc[
                        (scale_metrics_df['ML-algorithm'] == this_algo) & (scale_metrics_df['Workers'] == 1) &
                        (scale_metrics_df['Warmstart'] == False) & (scale_metrics_df['num_params'] <= dim_threshold), :]

        # High dimensional HP space (dimensionality > dim_threshold)
        elif dim_class == 'HighDimensional':

            # Filter according to the ML algorithm, the setup variables and the dimensionality
            sub_frame = scale_metrics_df.loc[
                        (scale_metrics_df['ML-algorithm'] == this_algo) & (scale_metrics_df['Workers'] == 1) &
                        (scale_metrics_df['Warmstart'] == False) & (scale_metrics_df['num_params'] > dim_threshold), :]

        else:
            raise Exception('Unknown dimensionality flag.')

        # Rank the HPO techniques according to their final performance
        sub_frame['final_rank'] = sub_frame.loc[:, 'Mean (final test loss)'].rank(axis=0, method='average',
                                                                                  na_option='bottom',
                                                                                  ascending=True)

        # Rank the HPO techniques according to their anytime performance
        sub_frame['anytime_rank'] = sub_frame.loc[:, 't outperform default [s]'].rank(axis=0, method='average',
                                                                                      na_option='bottom',
                                                                                      ascending=True)

        for this_tech in sub_frame['HPO-method'].unique():
            final_rank = sub_frame.loc[(sub_frame['HPO-method'] == this_tech), 'final_rank'].values[0]
            anytime_rank = sub_frame.loc[(sub_frame['HPO-method'] == this_tech), 'anytime_rank'].values[0]

            hpo_list.append(this_tech)
            ml_list.append(this_algo)
            final_rank_list.append(final_rank)
            anytime_rank_list.append(anytime_rank)

    # Create a pandas DataFrame to store the results
    scale_performance_df = pd.DataFrame({'HPO-method': hpo_list,
                                         'ML-algorithm': ml_list,
                                         'Final Performance Rank': final_rank_list,
                                         'Anytime Performance Rank': anytime_rank_list})

    # Compute the average rank for each HPO technique
    avg_final_rank_dict = {}
    avg_anytime_rank_dict = {}
    for this_tech in scale_performance_df['HPO-method'].unique():
        # Mean final rank of this HPO technique average over all ML algorithms
        avg_final_rank = np.nanmean(scale_performance_df.loc[scale_performance_df['HPO-method'] == this_tech,
                                                             'Final Performance Rank'].values)

        # Mean anytime rank of this HPO technique average over all ML algorithms
        avg_anytime_rank = np.nanmean(scale_performance_df.loc[scale_performance_df['HPO-method'] == this_tech,
                                                               'Anytime Performance Rank'].values)

        avg_final_rank_dict[this_tech] = avg_final_rank
        avg_anytime_rank_dict[this_tech] = avg_anytime_rank

    # Sort the dictionaries according to the rank achieved (descending)
    sorted_final_rank_dict = dict(sorted(avg_final_rank_dict.items(), key=lambda item: item[1], reverse=False))
    sorted_anytime_rank_dict = dict(sorted(avg_anytime_rank_dict.items(), key=lambda item: item[1], reverse=False))

    list_of_dicts = [sorted_final_rank_dict, sorted_anytime_rank_dict]
    bar_colors = ['#179c7d', '#0062a5']
    y_labels = ['Average Rank', 'Average Rank']
    performance_str = ['FinalPerformance', 'AnytimePerformance']

    # Bar plots to visualize the results
    for i in range(len(list_of_dicts)):
        fig, ax = plt.subplots(figsize=small_fig_size)
        plt.bar(x=list_of_dicts[i].keys(), height=list_of_dicts[i].values(), color=bar_colors[i], width=bar_width)

        # Formatting
        # ax.set_xlabel('HPO technique', fontweight='semibold', fontsize=font_size - 1, color='#969696',
        #               fontname=font_name)
        ax.set_ylabel(y_labels[i], fontweight='semibold', fontsize=font_size - 1, color='#969696', fontname=font_name)
        ax.spines['bottom'].set_color('#969696')
        ax.spines['top'].set_color('#969696')
        ax.spines['right'].set_color('#969696')
        ax.spines['left'].set_color('#969696')
        ax.tick_params(axis='x', colors='#969696', labelsize=font_size - 1, rotation=xtick_rotation)
        ax.tick_params(axis='y', colors='#969696', labelsize=font_size - 1)

        # Display the values in each bar
        for idx, val in enumerate(list_of_dicts[i].values()):
            ax.text(list(list_of_dicts[i].keys())[idx], 0.25, round(float(val), 2), color='white', ha='center',
                    va='bottom', fontsize=font_size - 1, fontname=font_name, rotation=90)

        fig_str = 'Scalability_' + dim_class + '_' + performance_str[i]

        if not os.path.isdir('./hpo_framework/results/' + dataset + '/Scalability/'):
            os.mkdir('./hpo_framework/results/' + dataset + '/Scalability/')

        fig_name1 = './hpo_framework/results/' + dataset + '/Scalability/' + dataset + fig_str + '.jpg'
        fig_name2 = './hpo_framework/results/' + dataset + '/Scalability/' + dataset + fig_str + '.svg'
        plt.savefig(fig_name1, bbox_inches='tight')
        plt.savefig(fig_name2, bbox_inches='tight')

        # TODO: Save results in table format

########################################################################################################################
# 5. Flexibility

complexity_classes = [1, 2, 3]

flex_metrics_df = metrics_df.copy(deep=True)

flex_metrics_df.loc[(flex_metrics_df['# cat. HPs'] > 0), 'Complexity Class'] = 3
flex_metrics_df.loc[(flex_metrics_df['# cat. HPs'] == 0) & (flex_metrics_df['# int. HPs'] > 0), 'Complexity Class'] = 2
flex_metrics_df.loc[(flex_metrics_df['# cat. HPs'] == 0) & (flex_metrics_df['# int. HPs'] == 0) &
                    (flex_metrics_df['# cont. HPs'] > 0), 'Complexity Class'] = 1

# Iterate over the different complexity classes
for cplx_class in complexity_classes:

    hpo_list = []
    ml_list = []
    final_rank_list = []
    anytime_rank_list = []

    # Iterate over ML algorithms
    for this_algo in ml_algorithms:

        # Filter according to the ML algorithm, the setup variables and the complexity class
        sub_frame = flex_metrics_df.loc[
                    (flex_metrics_df['ML-algorithm'] == this_algo) & (flex_metrics_df['Workers'] == 1) &
                    (flex_metrics_df['Warmstart'] == False) & (flex_metrics_df['Complexity Class'] == cplx_class), :]

        # Rank the HPO techniques according to their final performance
        sub_frame['final_rank'] = sub_frame.loc[:, 'Mean (final test loss)'].rank(axis=0, method='average',
                                                                                  na_option='bottom',
                                                                                  ascending=True)

        # Rank the HPO techniques according to their anytime performance
        sub_frame['anytime_rank'] = sub_frame.loc[:, 't outperform default [s]'].rank(axis=0, method='average',
                                                                                      na_option='bottom',
                                                                                      ascending=True)

        for this_tech in sub_frame['HPO-method'].unique():
            final_rank = sub_frame.loc[(sub_frame['HPO-method'] == this_tech), 'final_rank'].values[0]
            anytime_rank = sub_frame.loc[(sub_frame['HPO-method'] == this_tech), 'anytime_rank'].values[0]

            hpo_list.append(this_tech)
            ml_list.append(this_algo)
            final_rank_list.append(final_rank)
            anytime_rank_list.append(anytime_rank)

    # Create a pandas DataFrame to store the results
    flex_performance_df = pd.DataFrame({'HPO-method': hpo_list,
                                        'ML-algorithm': ml_list,
                                        'Final Performance Rank': final_rank_list,
                                        'Anytime Performance Rank': anytime_rank_list})

    # Compute the average rank for each HPO technique
    avg_final_rank_dict = {}
    avg_anytime_rank_dict = {}
    for this_tech in flex_performance_df['HPO-method'].unique():
        # Mean final rank of this HPO technique average over all ML algorithms
        avg_final_rank = np.nanmean(flex_performance_df.loc[flex_performance_df['HPO-method'] == this_tech,
                                                            'Final Performance Rank'].values)

        # Mean anytime rank of this HPO technique average over all ML algorithms
        avg_anytime_rank = np.nanmean(flex_performance_df.loc[flex_performance_df['HPO-method'] == this_tech,
                                                              'Anytime Performance Rank'].values)

        avg_final_rank_dict[this_tech] = avg_final_rank
        avg_anytime_rank_dict[this_tech] = avg_anytime_rank

    # Sort the dictionaries according to the rank achieved (descending)
    sorted_final_rank_dict = dict(sorted(avg_final_rank_dict.items(), key=lambda item: item[1], reverse=False))
    sorted_anytime_rank_dict = dict(sorted(avg_anytime_rank_dict.items(), key=lambda item: item[1], reverse=False))

    list_of_dicts = [sorted_final_rank_dict, sorted_anytime_rank_dict]
    bar_colors = ['#179c7d', '#0062a5']
    y_labels = ['Average Rank', 'Average Rank']
    performance_str = ['FinalPerformance', 'AnytimePerformance']

    # Bar plots to visualize the results
    for i in range(len(list_of_dicts)):
        fig, ax = plt.subplots(figsize=small_fig_size)
        plt.bar(x=list_of_dicts[i].keys(), height=list_of_dicts[i].values(), color=bar_colors[i], width=bar_width)

        # Formatting
        # ax.set_xlabel('HPO technique', fontweight='semibold', fontsize=font_size - 1, color='#969696',
        #               fontname=font_name)
        ax.set_ylabel(y_labels[i], fontweight='semibold', fontsize=font_size - 1, color='#969696', fontname=font_name)
        ax.spines['bottom'].set_color('#969696')
        ax.spines['top'].set_color('#969696')
        ax.spines['right'].set_color('#969696')
        ax.spines['left'].set_color('#969696')
        ax.tick_params(axis='x', colors='#969696', labelsize=font_size - 1, rotation=xtick_rotation)
        ax.tick_params(axis='y', colors='#969696', labelsize=font_size - 1)

        # Display the values in each bar
        for idx, val in enumerate(list_of_dicts[i].values()):
            ax.text(list(list_of_dicts[i].keys())[idx], 0.25, round(float(val), 2), color='white', ha='center',
                    va='bottom', fontsize=font_size - 1, fontname=font_name, rotation=90)

        fig_str = 'Flexibility' + str(cplx_class) + '_' + performance_str[i]

        if not os.path.isdir('./hpo_framework/results/' + dataset + '/Flexibility/'):
            os.mkdir('./hpo_framework/results/' + dataset + '/Flexibility/')

        fig_name1 = './hpo_framework/results/' + dataset + '/Flexibility/' + dataset + fig_str + '.jpg'
        fig_name2 = './hpo_framework/results/' + dataset + '/Flexibility/' + dataset + fig_str + '.svg'
        plt.savefig(fig_name1, bbox_inches='tight')
        plt.savefig(fig_name2, bbox_inches='tight')

        # TODO: Save results in table format

########################################################################################################################
# 2.1 Over-fitting

hpo_list = []
med_overfit_list = []

# Iterate over HPO techniques
for this_tech in hpo_techniques:
    # Filter for HPO-technique

    # # TODO: Integrate Fabolas again (extreme Overfitting on turbofan (SVR))
    # if this_tech == 'Fabolas':
    #     continue

    sub_frame = metrics_df.loc[(metrics_df['HPO-method'] == this_tech), :]

    sub_frame.loc[:, 'Overfitting [%]'] = sub_frame.loc[:, 'Generalization error'] / \
                                          sub_frame.loc[:, 'Mean (final validation loss)'] * 100

    med_over_fit = sub_frame.loc[:, 'Overfitting [%]'].median(axis=0, skipna=True)
    med_overfit_list.append(med_over_fit)
    hpo_list.append(this_tech)

# Create a pandas DataFrame to store the results
overfit_df = pd.DataFrame({'HPO-method': hpo_list,
                           'Median of Overfitting [%]': med_overfit_list
                           })

overfit_df.sort_values(by='Median of Overfitting [%]', axis=0, ascending=False, na_position='last', inplace=True)

# Bar plot to visualize the results
fig, ax = plt.subplots(figsize=large_fig_size)
# plt.barh(y=overfit_df['HPO-method'], width=overfit_df['Median of Overfitting [%]'], color='#0062a5', height=bar_width)
plt.bar(x=overfit_df['HPO-method'], height=overfit_df['Median of Overfitting [%]'], color='#0062a5', width=bar_width)

# Formatting
ax.set_xlabel('HPO technique', fontweight='semibold', fontsize='large', color='#969696')
ax.set_ylabel('Median of Overfitting [%]', fontweight='semibold', fontsize=font_size - 1, color='#969696',
              fontname=font_name)
ax.spines['bottom'].set_color('#969696')
ax.spines['top'].set_color('#969696')
ax.spines['right'].set_color('#969696')
ax.spines['left'].set_color('#969696')
ax.tick_params(axis='x', colors='#969696', rotation=xtick_rotation, labelsize=font_size - 1)
ax.tick_params(axis='y', colors='#969696', labelsize=font_size - 1)

# plt.yscale('log')

# Display the values in each bar
for idx, val in enumerate(overfit_df['Median of Overfitting [%]']):
    ax.text(list(overfit_df['HPO-method'])[idx], 0.1, round(float(val), 2), color='white', ha='center', va='bottom',
            fontweight='normal', fontsize=font_size - 1, fontname=font_name)

if not os.path.isdir('./hpo_framework/results/' + dataset + '/FinalPerformance/'):
    os.mkdir('./hpo_framework/results/' + dataset + '/FinalPerformance/')

fig_name1 = './hpo_framework/results/' + dataset + '/FinalPerformance/' + dataset + '_overfitting.jpg'
fig_name2 = './hpo_framework/results/' + dataset + '/FinalPerformance/' + dataset + '_overfitting.svg'
plt.savefig(fig_name1, bbox_inches='tight')
plt.savefig(fig_name2, bbox_inches='tight')

# TODO: Save results in table format

########################################################################################################################
# Create overview table

ml_algo_list = []
final_list = []
anytime_list = []
dim_list = []
cplx_list = []

for ml_algo in metrics_df['ML-algorithm'].unique():

    # Filter for ML algorithm
    sub_frame = metrics_df.loc[metrics_df['ML-algorithm'] == ml_algo, :]

    # Best HPO technique and setup regarding final performance
    best_final_idx = sub_frame['Mean (final test loss)'].idxmin(axis=0, skipna=True)
    best_final_tech = sub_frame.loc[best_final_idx, 'HPO-method']
    best_final_worker = sub_frame.loc[best_final_idx, 'Workers']
    best_final_wst = sub_frame.loc[best_final_idx, 'Warmstart']
    best_final_loss = sub_frame.loc[best_final_idx, 'Mean (final test loss)']

    # Best HPO technique and setup regarding anytime performance
    best_anytime_idx = sub_frame['t outperform default [s]'].idxmin(axis=0, skipna=True)
    best_anytime_tech = sub_frame.loc[best_anytime_idx, 'HPO-method']
    best_anytime_worker = sub_frame.loc[best_anytime_idx, 'Workers']
    best_anytime_wst = sub_frame.loc[best_anytime_idx, 'Warmstart']
    best_anytime_wall = sub_frame.loc[best_anytime_idx, 't outperform default [s]']

    # Determine complexity class and dimensionality of the configuration space
    num_cont_hps = list(sub_frame.loc[:, '# cont. HPs'])[0]
    num_int_hps = list(sub_frame.loc[:, '# int. HPs'])[0]
    num_cat_hps = list(sub_frame.loc[:, '# cat. HPs'])[0]

    # Check dimensionality
    n_hps = num_cont_hps + num_int_hps + num_cat_hps  # total number of Hyperparameters
    if n_hps <= dim_threshold:
        this_dim = 'LowDimensional'
    else:
        this_dim = 'HighDimensional'

    # Check complexity class
    if num_cat_hps > 0:
        cplx_class = 3
    elif num_int_hps > 0:
        cplx_class = 2
    else:
        cplx_class = 1

    # Append results to the lists
    ml_algo_list.append(ml_algo)
    final_list.append((best_final_loss, best_final_tech, best_final_worker, best_final_wst))
    anytime_list.append((best_anytime_wall, best_anytime_tech, best_anytime_worker, best_anytime_wst))
    dim_list.append(this_dim)
    cplx_list.append(cplx_class)

# create pd.DataFrame to store the results
overview_df = pd.DataFrame({
    'ML-algorithm': ml_algo_list,
    'Final Performance': final_list,
    'Anytime Performance': anytime_list,
    'Dimensionality': dim_list,
    'Complexity Class': cplx_list
})

if not os.path.isdir('./hpo_framework/results/' + dataset + '/Overview/'):
    os.mkdir('./hpo_framework/results/' + dataset + '/Overview/')

# Save DataFrame to .csv File
table_name = './hpo_framework/results/' + dataset + '/Overview/' + dataset + '_overview_table.csv'
overview_df.to_csv(table_name)

########################################################################################################################
# Assess the effects of parallelization and warm start on final and anytime performance

algo_list = []
hpo_list = []
effect_final_parallel_lst = []
effect_any_parallel_lst = []
effect_final_wst_lst = []
effect_any_wst_lst = []

# Iterate over ML algorithms
for this_algo in metrics_df['ML-algorithm'].unique():

    # TODO: How to deal with Keras (no parallel setup)
    if this_algo == 'KerasRegressor' or this_algo == 'KerasClassifier':
        continue

    # Iterate over HPO techniques
    for this_tech in metrics_df['HPO-method'].unique():

        # TODO: Add 8 worker metrics for Fabolas and Bohamiann
        if this_tech == 'Fabolas' or this_tech == 'Bohamiann':
            continue

        # Filter for ML algorithm and HPO technique
        sub_frame = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) &
                                   (metrics_df['HPO-method'] == this_tech), :]

        # Setup 1: single worker, no warm start
        final_single_nowst = sub_frame.loc[(sub_frame['Workers'] == 1) & (sub_frame['Warmstart'] == False),
                                           'Mean (final test loss)'].values[0]

        anytime_single_nowst = sub_frame.loc[(sub_frame['Workers'] == 1) & (sub_frame['Warmstart'] == False),
                                             't outperform default [s]'].values[0]

        # Setup 2: 8 parallel workers, no warm start
        final_parallel_nowst = sub_frame.loc[(sub_frame['Workers'] == 8) & (sub_frame['Warmstart'] == False),
                                             'Mean (final test loss)'].values[0]

        anytime_parallel_nowst = sub_frame.loc[(sub_frame['Workers'] == 8) & (sub_frame['Warmstart'] == False),
                                               't outperform default [s]'].values[0]

        # Setup 3: single worker, warm start
        final_single_wst = sub_frame.loc[(sub_frame['Workers'] == 1) & (sub_frame['Warmstart'] == True),
                                         'Mean (final test loss)'].values[0]

        anytime_single_wst = sub_frame.loc[(sub_frame['Workers'] == 1) & (sub_frame['Warmstart'] == True),
                                           't outperform default [s]'].values[0]

        # Calculate the effects of parallelization and warm start on final and anytime performance

        # 1. Relative effect of parallelization on final performance (decrease is good)
        para_final_effect = 100 * (final_parallel_nowst - final_single_nowst) / final_single_nowst

        # 2. Relative effect of parallelization on anytime performance (decrease is good)
        para_any_effect = 100 * (anytime_parallel_nowst - anytime_single_nowst) / anytime_single_nowst

        # 3. Relative effect of warm start on final performance (decrease is good)
        wst_final_effect = 100 * (final_single_wst - final_single_nowst) / final_single_nowst

        # 4. Relative effect of warm start on anytime performance (decrease is good)
        wst_any_effect = 100 * (anytime_single_wst - anytime_single_nowst) / anytime_single_nowst

        # Append results
        algo_list.append(this_algo)
        hpo_list.append(this_tech)
        effect_final_parallel_lst.append(para_final_effect)
        effect_any_parallel_lst.append(para_any_effect)
        effect_final_wst_lst.append(wst_final_effect)
        effect_any_wst_lst.append(wst_any_effect)

para_wst_effect_df = pd.DataFrame({
    'ML-algorithm': algo_list,
    'HPO-method': hpo_list,
    'Effect Parallelization (Final Performance)': effect_final_parallel_lst,
    'Effect Parallelization (Anytime Performance)': effect_any_parallel_lst,
    'Effect Warm Start (Final Performance)': effect_final_wst_lst,
    'Effect Warm Start (Anytime Performance)': effect_any_wst_lst
})

# Compute the average effects for each HPO technique
avg_effect_final_parallel_lst = []
avg_effect_any_parallel_lst = []
avg_effect_final_wst_lst = []
avg_effect_any_wst_lst = []
for this_tech in hpo_list:
    # Compute the average effects
    avg_final_parallel = para_wst_effect_df.loc[para_wst_effect_df['HPO-method'] == this_tech,
                                                'Effect Parallelization (Final Performance)'].mean(axis=0, skipna=True)
    avg_any_parallel = para_wst_effect_df.loc[para_wst_effect_df['HPO-method'] == this_tech,
                                              'Effect Parallelization (Anytime Performance)'].mean(axis=0, skipna=True)
    avg_final_wst = para_wst_effect_df.loc[para_wst_effect_df['HPO-method'] == this_tech,
                                           'Effect Warm Start (Final Performance)'].mean(axis=0, skipna=True)
    avg_any_wst = para_wst_effect_df.loc[para_wst_effect_df['HPO-method'] == this_tech,
                                         'Effect Warm Start (Anytime Performance)'].mean(axis=0, skipna=True)

    # Append the results
    avg_effect_final_parallel_lst.append(avg_final_parallel)
    avg_effect_any_parallel_lst.append(avg_any_parallel)
    avg_effect_final_wst_lst.append(avg_final_wst)
    avg_effect_any_wst_lst.append(avg_any_wst)

# Create pd.DataFrame for plotting
plot_df = pd.DataFrame({
    'HPO-method': hpo_list,
    'Avg. Effect Parallelization (Final Performance)': avg_effect_final_parallel_lst,
    'Avg. Effect Parallelization (Anytime Performance)': avg_effect_any_parallel_lst,
    'Avg. Effect Warm Start (Final Performance)': avg_effect_final_wst_lst,
    'Avg. Effect Warm Start (Anytime Performance)': avg_effect_any_wst_lst
})

# Scatter plots
final_fig, final_ax = plt.subplots(figsize=large_fig_size)
any_fig, any_ax = plt.subplots(figsize=large_fig_size)

final_para_sct = final_ax.scatter(x=plot_df['HPO-method'],
                                  y=plot_df['Avg. Effect Parallelization (Final Performance)'],
                                  c='#179c7d')

final_wst_sct = final_ax.scatter(x=plot_df['HPO-method'],
                                 y=plot_df['Avg. Effect Warm Start (Final Performance)'],
                                 c='#ff6600')

any_para_sct = any_ax.scatter(x=plot_df['HPO-method'],
                              y=plot_df['Avg. Effect Parallelization (Anytime Performance)'],
                              c='#179c7d')

any_wst_sct = any_ax.scatter(x=plot_df['HPO-method'],
                             y=plot_df['Avg. Effect Warm Start (Anytime Performance)'],
                             c='#ff6600')

# Formatting
# X labels
final_ax.set_xlabel('HPO technique', fontweight='semibold', fontsize=font_size, color='#969696',
                    fontname=font_name)
any_ax.set_xlabel('HPO technique', fontweight='semibold', fontsize=font_size, color='#969696',
                  fontname=font_name)

# Y labels
final_ax.set_ylabel('Relative effect on final test loss [%]', fontweight='semibold', fontsize=font_size,
                    color='#969696', fontname=font_name)

any_ax.set_ylabel('Relative effect on required time to outperform default [%]', fontweight='semibold',
                  fontsize=font_size,
                  color='#969696', fontname=font_name)

# Tick params
for ax in [final_ax, any_ax]:
    ax.spines['bottom'].set_color('#969696')
    ax.spines['top'].set_color('#969696')
    ax.spines['right'].set_color('#969696')
    ax.spines['left'].set_color('#969696')
    ax.tick_params(axis='x', colors='#969696', rotation=45)
    ax.tick_params(axis='y', colors='#969696')

# Add legend
legend_labels = ('Parallelization (1 vs. 8 workers)', 'Warm start')
final_ax.legend((final_para_sct, final_wst_sct), legend_labels)
any_ax.legend((any_para_sct, any_wst_sct), legend_labels)

# Save figures
fig_name_final1 = './hpo_framework/results/' + dataset + '/Overview/' + dataset + '_para_wst_final_performance.jpg'
fig_name_final2 = './hpo_framework/results/' + dataset + '/Overview/' + dataset + '_para_wst_final_performance.svg'
final_fig.savefig(fig_name_final1, bbox_inches='tight')
final_fig.savefig(fig_name_final2, bbox_inches='tight')

fig_name_any1 = './hpo_framework/results/' + dataset + '/Overview/' + dataset + '_para_wst_anytime_performance.jpg'
fig_name_any2 = './hpo_framework/results/' + dataset + '/Overview/' + dataset + '_para_wst_anytime_performance.svg'
any_fig.savefig(fig_name_any1, bbox_inches='tight')
any_fig.savefig(fig_name_any2, bbox_inches='tight')

# Boxplots
box_plots = ['Effect Parallelization (Final Performance)', 'Effect Parallelization (Anytime Performance)',
             'Effect Warm Start (Final Performance)', 'Effect Warm Start (Anytime Performance)']

y_labels = ['Impact on final test loss [%]',
            'Impact on time to outperform baseline [%]',
            'Impact on final test loss [%]',
            'Impact on time to outperform baseline [%]']

fig_names = ['_box_para_final',
             '_box_para_any',
             '_box_wst_final',
             '_box_wst_any']

for i in range(len(box_plots)):

    this_plot = box_plots[i]
    this_label = y_labels[i]

    box_label_list = []
    y_list = []
    for j in range(len(para_wst_effect_df['HPO-method'].unique())):
        this_tech = list(para_wst_effect_df['HPO-method'].unique())[j]
        box_label_list.append(this_tech)

        values = para_wst_effect_df.loc[para_wst_effect_df['HPO-method'] == this_tech,
                                        this_plot].values
        values = values[~np.isnan(values)]
        y_list.append(values)

    box_fig, box_ax = plt.subplots(figsize=small_fig_size)
    box_ax.boxplot(x=y_list, labels=box_label_list, showfliers=False)
    box_ax.hlines(y=0.0, xmin=1, xmax=len(box_label_list), linestyles='dashed', colors='deepskyblue')

    box_ax.set_ylabel(this_label, fontweight='semibold', fontsize=font_size,
                      color='#969696', fontname=font_name)

    box_ax.spines['bottom'].set_color('#969696')
    box_ax.spines['top'].set_color('#969696')
    box_ax.spines['right'].set_color('#969696')
    box_ax.spines['left'].set_color('#969696')
    box_ax.tick_params(axis='x', colors='#969696', rotation=90)
    box_ax.tick_params(axis='y', colors='#969696')

    fig_name1 = './hpo_framework/results/' + dataset + '/Overview/' + dataset + fig_names[i] + '.jpg'
    fig_name2 = './hpo_framework/results/' + dataset + '/Overview/' + dataset + fig_names[i] + '.svg'
    box_fig.savefig(fig_name1, bbox_inches='tight')
    box_fig.savefig(fig_name2, bbox_inches='tight')
