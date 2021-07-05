import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

# Create plot on dataset or benchmarking study level
aggregation_level = 'benchmarking'  # 'benchmarking', 'dataset'

datasets = ['turbofan', 'scania', 'sensor', 'blisk', 'surface']

# Specify the dataset, if the heatmap is created on dataset level
dataset = 'surface'

cut_variants = ['All Time Budgets', 'Max Time Budget', 'Low Time Budgets']

ipt_colors = ['#179c7d', '#438cd4', '#ffcc99']
bar_width = 0.3

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

os.chdir('..')

if aggregation_level == 'dataset':

    file_path = './hpo_framework/results/%s/RankingAnalysis/%s_ranked_metrics.csv' % (
        dataset, dataset)

    ranked_df = pd.read_csv(file_path, index_col=0)

elif aggregation_level == 'benchmarking':

    j = 0

    for this_dataset in datasets:

        file_path = './hpo_framework/results/%s/RankingAnalysis/%s_ranked_metrics.csv' % (
            this_dataset, this_dataset)

        if j == 0:

            ranked_df = pd.read_csv(file_path, index_col=0)

        else:

            this_df = pd.read_csv(file_path, index_col=0)

            ranked_df = pd.concat(
                objs=[ranked_df, this_df], axis=0, ignore_index=True)

        j += 1

    ranked_df.reset_index(drop=True, inplace=True)

else:

    raise Exception('Unknown aggregation level!')

hpo_techs = ['Default HPs', 'RandomSearch', 'GPBO', 'SMAC', 'TPE', 'CMA-ES',
             'Hyperband', 'BOHB', 'Fabolas', 'Bohamiann']

dv_per_cut_df = pd.DataFrame([])

idx = 0

for i in range(len(cut_variants)):

    this_cut = cut_variants[i]

    if this_cut == 'All Time Budgets':

        cut_df = ranked_df.loc[(ranked_df['1st metric'] == 'Max Cut Validation Loss') |
                               (ranked_df['1st metric'] == '2nd Cut Validation Loss') |
                               (ranked_df['1st metric'] == '3rd Cut Validation Loss') |
                               (ranked_df['1st metric'] == '4th Cut Validation Loss'), :]

    elif this_cut == 'Max Time Budget':

        cut_df = ranked_df.loc[(ranked_df['1st metric']
                                == 'Max Cut Validation Loss'), :]

    elif this_cut == 'Low Time Budgets':

        cut_df = ranked_df.loc[(ranked_df['1st metric'] == '2nd Cut Validation Loss') |
                               (ranked_df['1st metric'] == '3rd Cut Validation Loss') |
                               (ranked_df['1st metric'] == '4th Cut Validation Loss'), :]

    else:

        raise Exception('Unknown time budget cut!')

    for this_tech in hpo_techs:

        this_distance_value = np.nanmean(cut_df.loc[(cut_df['1st metric HPO-technique'] == this_tech),
                                                    '1st metric scaled deviation'].to_numpy())

        dv_per_cut_df.loc[idx, 'HPO technique'] = hpo_map_dict[this_tech]
        dv_per_cut_df.loc[idx,
                          'Average Distance Value'] = this_distance_value
        dv_per_cut_df.loc[idx, 'Cut Variant'] = cut_variants[i]

        idx += 1

sns.set_theme(font='Arial', style='ticks')

cat = sns.catplot(data=dv_per_cut_df, x='HPO technique', y='Average Distance Value',
                  hue='Cut Variant', kind='bar', legend=False,
                  palette=sns.color_palette(['#3E927F', '#8bcdbe', '#dddddd']))

cat.fig.set_size_inches(6.5, 4)
cat.set(xlabel=None)

plt.tick_params(axis='both', labelsize=11)
plt.tick_params(axis='x', rotation=90)
plt.legend(loc='upper right')

if aggregation_level == 'dataset':

    fig_name = './hpo_framework/results/%s/RankingAnalysis/dv_per_cut_variant_%s.svg' % (
        dataset, dataset)

elif aggregation_level == 'benchmarking':

    fig_name = './hpo_framework/results/dv_per_cut_variant_overall.svg'

else:

    raise Exception('Unknown aggregation level!')

cat.fig.savefig(fig_name, bbox_inches='tight')
