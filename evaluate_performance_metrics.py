import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

dataset = 'turbofan'

metrics_path = './hpo_framework/results/' + dataset + '/metrics_' + dataset + '.csv'
metrics_df = pd.read_csv(metrics_path, index_col=0)

hpo_techniques = metrics_df['HPO-method'].unique()
ml_algorithms = metrics_df['ML-algorithm'].unique()

# Plot params
bar_width = 0.6

########################################################################################################################
# 1. Effective use of parallel resources
hpo_list = []
ml_list = []
speed_up_list = []

# Iterate over ML algorithms
for this_algo in ml_algorithms:

    # TODO: How to deal with Keras (no parallel setup)
    if this_algo == 'KerasRegressor':
        continue

    # Iterate over HPO techniques
    for this_tech in hpo_techniques:

        # TODO: Add 8 WORKER METRICS FOR FABOLAS
        if this_tech == 'Fabolas' or this_tech == 'Bohamiann':
            continue

        # Filter for ML algorithm, HPO technique and no warm start
        sub_frame = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) & (metrics_df['HPO-method'] == this_tech)
                                   & (metrics_df['Warmstart'] == False), :]

        # Wall clock times for single and multiple worker setup
        single_worker_time = sub_frame.loc[sub_frame['Workers'] == 1, 'Wall clock time [s]'].values
        multiple_worker_time = sub_frame.loc[sub_frame['Workers'] == 8, 'Wall clock time [s]'].values

        # Calculate the speed up
        speed_up_factor = float(single_worker_time / multiple_worker_time)

        # Append the results
        hpo_list.append(this_tech)
        ml_list.append(this_algo)
        speed_up_list.append(speed_up_factor)

# Create a pandas DataFrame to store the results
speed_up_df = pd.DataFrame({'HPO-method': hpo_list,
                            'ML-algorithm': ml_list,
                            'Speed-up': speed_up_list})

# Compute the average speed up, that each HPO technique achieves due to parallelization
avg_speed_up_dict = {}
for this_tech in speed_up_df['HPO-method'].unique():
    this_avg = np.nanmean(speed_up_df.loc[speed_up_df['HPO-method'] == this_tech, 'Speed-up'].values)
    avg_speed_up_dict[this_tech] = this_avg

# Sort the dictionary according to the speed up achieved (descending)
sorted_avg_speed_up_dict = dict(sorted(avg_speed_up_dict.items(), key=lambda item: item[1], reverse=True))

# Bar plot to visualize the results
fig, ax = plt.subplots(figsize=(11, 9))
plt.bar(x=sorted_avg_speed_up_dict.keys(), height=sorted_avg_speed_up_dict.values(), color='#179c7d', width=bar_width)

# Formatting
ax.set_xlabel('HPO technique', fontweight='semibold', fontsize='large', color='#969696')
ax.set_ylabel('Speed up factor', fontweight='semibold', fontsize='large', color='#969696')
ax.spines['bottom'].set_color('#969696')
ax.spines['top'].set_color('#969696')
ax.spines['right'].set_color('#969696')
ax.spines['left'].set_color('#969696')
ax.tick_params(axis='x', colors='#969696')
ax.tick_params(axis='y', colors='#969696')

# Display the values in each bar
for idx, val in enumerate(sorted_avg_speed_up_dict.values()):
    ax.text(list(sorted_avg_speed_up_dict.keys())[idx], 0.1, round(float(val), 2), color='white', ha='center',
            fontweight='semibold', fontsize='large')

if not os.path.isdir('./hpo_framework/results/' + dataset + '/Parallelization/'):
    os.mkdir('./hpo_framework/results/' + dataset + '/Parallelization/')

fig_name1 = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_parallelization.jpg'
fig_name2 = './hpo_framework/results/' + dataset + '/Parallelization/' + dataset + '_parallelization.svg'
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

fig, ax = plt.subplots(figsize=(11, 9))
plt.bar(x=sorted_crash_dict.keys(), height=sorted_crash_dict.values(), color='#179c7d', width=bar_width)

# Formatting
ax.set_xlabel('HPO technique', fontweight='semibold', fontsize='large', color='#969696')
ax.set_ylabel('Share of crashed runs [%]', fontweight='semibold', fontsize='large', color='#969696')
ax.spines['bottom'].set_color('#969696')
ax.spines['top'].set_color('#969696')
ax.spines['right'].set_color('#969696')
ax.spines['left'].set_color('#969696')
ax.tick_params(axis='x', colors='#969696')
ax.tick_params(axis='y', colors='#969696')

# Display the values in each bar
for idx, val in enumerate(sorted_crash_dict.values()):
    ax.text(list(sorted_crash_dict.keys())[idx], .8, round(float(val), 2), color='white', ha='center',
            fontweight='semibold', fontsize='large')

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
    fig, ax = plt.subplots(figsize=(11, 9))
    plt.bar(x=sorted_rank_dict.keys(), height=sorted_rank_dict.values(), color='#179c7d', width=bar_width)

    # Formatting
    ax.set_xlabel('HPO technique', fontweight='semibold', fontsize='large', color='#969696')
    ax.set_ylabel('Average Rank (Final Performance)', fontweight='semibold', fontsize='large', color='#969696')
    ax.spines['bottom'].set_color('#969696')
    ax.spines['top'].set_color('#969696')
    ax.spines['right'].set_color('#969696')
    ax.spines['left'].set_color('#969696')
    ax.tick_params(axis='x', colors='#969696')
    ax.tick_params(axis='y', colors='#969696')

    # Display the values in each bar
    for idx, val in enumerate(sorted_rank_dict.values()):
        ax.text(list(sorted_rank_dict.keys())[idx], 0.1, round(float(val), 2), color='white', ha='center',
                fontweight='semibold', fontsize='large')

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

    # TODO: Save results in table format

    # TODO: Bar plot for over-fitting

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
    fig, ax = plt.subplots(figsize=(11, 9))
    plt.bar(x=sorted_rank_dict.keys(), height=sorted_rank_dict.values(), color='#179c7d', width=bar_width)

    # Formatting
    ax.set_xlabel('HPO technique', fontweight='semibold', fontsize='large', color='#969696')
    ax.set_ylabel('Average Rank (Anytime Performance)', fontweight='semibold', fontsize='large', color='#969696')
    ax.spines['bottom'].set_color('#969696')
    ax.spines['top'].set_color('#969696')
    ax.spines['right'].set_color('#969696')
    ax.spines['left'].set_color('#969696')
    ax.tick_params(axis='x', colors='#969696')
    ax.tick_params(axis='y', colors='#969696')

    # Display the values in each bar
    for idx, val in enumerate(sorted_rank_dict.values()):
        ax.text(list(sorted_rank_dict.keys())[idx], 0.1, round(float(val), 2), color='white', ha='center',
                fontweight='semibold', fontsize='large')

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
    y_labels = ['Average Rank (Final Performance)', 'Average Rank (Anytime Performance)']
    performance_str = ['FinalPerformance', 'AnytimePerformance']

    # Bar plots to visualize the results
    for i in range(len(list_of_dicts)):
        fig, ax = plt.subplots(figsize=(11, 9))
        plt.bar(x=list_of_dicts[i].keys(), height=list_of_dicts[i].values(), color=bar_colors[i], width=bar_width)

        # Formatting
        ax.set_xlabel('HPO technique', fontweight='semibold', fontsize='large', color='#969696')
        ax.set_ylabel(y_labels[i], fontweight='semibold', fontsize='large', color='#969696')
        ax.spines['bottom'].set_color('#969696')
        ax.spines['top'].set_color('#969696')
        ax.spines['right'].set_color('#969696')
        ax.spines['left'].set_color('#969696')
        ax.tick_params(axis='x', colors='#969696')
        ax.tick_params(axis='y', colors='#969696')

        # Display the values in each bar
        for idx, val in enumerate(list_of_dicts[i].values()):
            ax.text(list(list_of_dicts[i].keys())[idx], 0.1, round(float(val), 2), color='white', ha='center',
                    fontweight='semibold', fontsize='large')

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
    y_labels = ['Average Rank (Final Performance)', 'Average Rank (Anytime Performance)']
    performance_str = ['FinalPerformance', 'AnytimePerformance']

    # Bar plots to visualize the results
    for i in range(len(list_of_dicts)):
        fig, ax = plt.subplots(figsize=(11, 9))
        plt.bar(x=list_of_dicts[i].keys(), height=list_of_dicts[i].values(), color=bar_colors[i], width=bar_width)

        # Formatting
        ax.set_xlabel('HPO technique', fontweight='semibold', fontsize='large', color='#969696')
        ax.set_ylabel(y_labels[i], fontweight='semibold', fontsize='large', color='#969696')
        ax.spines['bottom'].set_color('#969696')
        ax.spines['top'].set_color('#969696')
        ax.spines['right'].set_color('#969696')
        ax.spines['left'].set_color('#969696')
        ax.tick_params(axis='x', colors='#969696')
        ax.tick_params(axis='y', colors='#969696')

        # Display the values in each bar
        for idx, val in enumerate(list_of_dicts[i].values()):
            ax.text(list(list_of_dicts[i].keys())[idx], 0.1, round(float(val), 2), color='white', ha='center',
                    fontweight='semibold', fontsize='large')

        fig_str = 'Flexibility' + str(cplx_class) + '_' + performance_str[i]

        if not os.path.isdir('./hpo_framework/results/' + dataset + '/Flexibility/'):
            os.mkdir('./hpo_framework/results/' + dataset + '/Flexibility/')

        fig_name1 = './hpo_framework/results/' + dataset + '/Flexibility/' + dataset + fig_str + '.jpg'
        fig_name2 = './hpo_framework/results/' + dataset + '/Flexibility/' + dataset + fig_str + '.svg'
        plt.savefig(fig_name1, bbox_inches='tight')
        plt.savefig(fig_name2, bbox_inches='tight')

        # TODO: Save results in table format
