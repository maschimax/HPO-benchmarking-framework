import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = 'turbofan'

metrics_path = './hpo_framework/results/' + dataset + '/metrics_' + dataset + '.csv'
metrics_df = pd.read_csv(metrics_path, index_col=0)

hpo_techniques = metrics_df['HPO-method'].unique()
ml_algorithms = metrics_df['ML-algorithm'].unique()

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
plt.bar(x=sorted_avg_speed_up_dict.keys(), height=sorted_avg_speed_up_dict.values(), color='#5cbaa4')

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

fig_name1 = './hpo_framework/results/' + dataset + '/' + dataset + '_parallelization.jpg'
fig_name2 = './hpo_framework/results/' + dataset + '/' + dataset + '_parallelization.svg'
plt.savefig(fig_name1, bbox_inches='tight')
plt.savefig(fig_name2, bbox_inches='tight')

plt.show()

# TODO: Save results in table format

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
plt.bar(x=sorted_crash_dict.keys(), height=sorted_crash_dict.values(), color='#5cbaa4')

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

fig_name1 = './hpo_framework/results/' + dataset + '/' + dataset + '_robustness.jpg'
fig_name2 = './hpo_framework/results/' + dataset + '/' + dataset + '_robustness.svg'
plt.savefig(fig_name1, bbox_inches='tight')
plt.savefig(fig_name2, bbox_inches='tight')

plt.show()

########################################################################################################################
# 2. Final Performance

# 2.1 Single worker, no warm start
