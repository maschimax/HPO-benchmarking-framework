import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = 'turbofan'

metrics_path = './hpo_framework/results/' + dataset + '/metrics_' + dataset + '.csv'
metrics_df = pd.read_csv(metrics_path, index_col=0)

hpo_techniques = metrics_df['HPO-method'].unique()
ml_algorithms = metrics_df['ML-algorithm'].unique()

# 1. Effective use of parallel resources

# Iterate over ML algorithms
hpo_list = []
ml_list = []
speed_up_list = []

for this_algo in ml_algorithms:

    # TODO: How to deal with Keras (no parallel setup)
    if this_algo == 'KerasRegressor':
        continue

    for this_tech in hpo_techniques:

        # TODO: Add 8 WORKER METRICS FOR FABOLAS
        if this_tech == 'Fabolas' or this_tech == 'Bohamiann':
            continue

        sub_frame = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) & (metrics_df['HPO-method'] == this_tech)
                                   & (metrics_df['Warmstart'] == False), :]

        single_worker_time = sub_frame.loc[sub_frame['Workers'] == 1, 'Wall clock time [s]'].values
        multiple_worker_time = sub_frame.loc[sub_frame['Workers'] == 8, 'Wall clock time [s]'].values

        speed_up_factor = float(single_worker_time / multiple_worker_time)

        hpo_list.append(this_tech)
        ml_list.append(this_algo)
        speed_up_list.append(speed_up_factor)

speed_up_df = pd.DataFrame({'HPO-method': hpo_list,
                            'ML-algorithm': ml_list,
                            'Speed-up': speed_up_list})

avg_speed_up_dict = {}
for this_tech in speed_up_df['HPO-method'].unique():
    this_avg = np.nanmean(speed_up_df.loc[speed_up_df['HPO-method'] == this_tech, 'Speed-up'].values)
    avg_speed_up_dict[this_tech] = this_avg

fig, ax = plt.subplots(figsize=(11, 9))
plt.bar(x=avg_speed_up_dict.keys(), height=avg_speed_up_dict.values(), color='#5cbaa4')
ax.set_xlabel('HPO-techniques', fontweight='semibold', fontsize='large', color='#969696')
ax.set_ylabel('Speed up factor', fontweight='semibold', fontsize='large', color='#969696')

ax.spines['bottom'].set_color('#969696')
ax.spines['top'].set_color('#969696')
ax.spines['right'].set_color('#969696')
ax.spines['left'].set_color('#969696')
ax.tick_params(axis='x', colors='#969696')
ax.tick_params(axis='y', colors='#969696')

for idx, val in enumerate(avg_speed_up_dict.values()):
    ax.text(list(avg_speed_up_dict.keys())[idx], 0.1, round(float(val), 2), color='white', ha='center',
            fontweight='semibold', fontsize='large')

plt.show()

bla = 0
