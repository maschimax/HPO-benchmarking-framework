import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

hpo_list = []
ml_list = []
speed_up_list = []
time_per_eval_list = []

datasets = ['turbofan', 'scania', 'sensor', 'blisk', 'surface']

hpo_map_dict = {
    'Bohamiann': 'BOHAMIANN',
    'BOHB': 'BOHB',
    'CMA-ES': 'CMA-ES',
    'Fabolas': 'FABOLAS',
    'GPBO': 'GPBO',
    'Hyperband': 'HB',
    'RandomSearch': 'Random Search',
    'SMAC': 'SMAC',
    'TPE': 'TPE',
    'Default HPs': 'Default HPs'
}

speed_up_df = pd.DataFrame([])

idx = 0

os.chdir('..')

for this_dataset in datasets:

    file_path = './hpo_framework/results/%s/metrics_%s.csv' % (
        this_dataset, this_dataset)

    metrics_df = pd.read_csv(file_path, index_col=0)

    ml_algos = metrics_df['ML-algorithm'].unique()
    hpo_techs = metrics_df['HPO-method'].unique()

    for this_algo in ml_algos:

        # No parallelized experiments for Keras
        if this_algo == 'KerasRegressor' or this_algo == 'KerasClassifier':
            continue

        for this_tech in hpo_techs:

            # No parallelization for Fabolas and Bohamiann
            if this_tech == 'Fabolas' or this_tech == 'Bohamiann':
                continue

            # Filter for ML algorithm, HPO technique and no warm start
            sub_frame = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) & (metrics_df['HPO-method'] == this_tech)
                                       & (metrics_df['Warmstart'] == False), :]

            rs_single_time = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) &
                                            (metrics_df['HPO-method'] == 'RandomSearch') &
                                            (metrics_df['Warmstart'] == False) &
                                            (metrics_df['Workers'] == 1), 'Wall clock time [s]'].values

            rs_evals = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) &
                                      (metrics_df['HPO-method'] == 'RandomSearch') &
                                      (metrics_df['Warmstart'] == False) &
                                      (metrics_df['Workers'] == 1), 'Evaluations'].values

            # Average time per eval (for Random Search on single worker setup w/o warm-start)
            if len(rs_single_time) != 0 and len(rs_evals) != 0:
                rs_time_per_eval = float(rs_single_time / rs_evals)
            else:
                rs_time_per_eval = np.nan

            # Wall clock times for single and multiple worker setup
            single_worker_time = sub_frame.loc[sub_frame['Workers']
                                               == 1, 'Wall clock time [s]'].values
            multiple_worker_time = sub_frame.loc[sub_frame['Workers']
                                                 == 8, 'Wall clock time [s]'].values

            if len(single_worker_time) == 0 or len(multiple_worker_time) == 0:
                continue

            this_speed_up = float(single_worker_time / multiple_worker_time)

            speed_up_df.loc[idx, 'Dataset'] = this_dataset
            speed_up_df.loc[idx, 'HPO-technique'] = this_tech
            speed_up_df.loc[idx, 'ML-algorithm'] = this_algo
            speed_up_df.loc[idx, 'Speed up'] = this_speed_up
            speed_up_df.loc[idx, 'Time per eval. [RS]'] = rs_time_per_eval

            idx += 1

# Boxplots to visualize the distribution of speed up factors for each HPO technique
box_labels = []
box_vec = []

for this_tech in speed_up_df['HPO-technique'].unique():

    box_labels.append(hpo_map_dict[this_tech])

    speed_up_factors = speed_up_df.loc[speed_up_df['HPO-technique']
                                       == this_tech, 'Speed up'].values

    speed_up_factors = speed_up_factors[~np.isnan(speed_up_factors)]
    box_vec.append(list(speed_up_factors))


box_fig, box_ax = plt.subplots(figsize=(7, 4))

box_ax.boxplot(x=box_vec, labels=box_labels, showfliers=True)
box_ax.hlines(y=1.0, xmin=1, xmax=len(box_labels),
              linestyles='dashed', colors='#969696')

box_ax.set_ylabel('Speed up', fontsize=11, fontname='Arial')
box_ax.tick_params(axis='x', rotation=90)

fig_name = './hpo_framework/results/speed_up_boxplot.svg'
plt.savefig(fig_name, bbox_inches='tight')

# Scatter plot to visualize the distribution of speed up factors over time

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

fig, ax = plt.subplots(figsize=(7, 4))

for this_tech in ['RandomSearch', 'TPE', 'CMA-ES']:

    sub_df = speed_up_df.loc[(speed_up_df['HPO-technique'] == this_tech), :]

    # ax.scatter(x=sub_df['Time per eval. [RS]'], y=sub_df['Speed up'], c=colors[this_tech])

    ax = sns.regplot(x='Time per eval. [RS]', y='Speed up', data=sub_df,
                     color=colors[this_tech], label=hpo_map_dict[this_tech], order=1)

ax.legend(fontsize=10)
ax.set_xlabel('Average training time [s]', fontsize=11, fontname='Arial')
ax.set_ylabel('Speed up', fontsize=11, fontname='Arial')
plt.xscale('log')

fig_name = './hpo_framework/results/speed_up_over_eval_time.svg'
plt.savefig(fig_name, bbox_inches='tight')
