import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

dataset = 'turbofan'
setup_variants = [(1, False), (8, False), (1, True)]

setup_labels = ['Single worker, no warm-start',
                'Eight workers, no warm-start', 'Single worker, warm-start']

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

file_path = './hpo_framework/results/%s/RankingAnalysis/%s_ranked_metrics.csv' % (
    dataset, dataset)

ranked_df = pd.read_csv(file_path, index_col=0)
hpo_techs = ['Default HPs', 'RandomSearch', 'GPBO', 'SMAC', 'TPE', 'CMA-ES',
             'Hyperband', 'BOHB', 'Fabolas', 'Bohamiann']

dv_per_setup_df = pd.DataFrame([])

idx = 0

for i in range(len(setup_variants)):

    this_setup = setup_variants[i]

    setup_df = ranked_df.loc[(ranked_df['Workers'] == this_setup[0]) &
                             (ranked_df['Warm start'] == this_setup[1]) & (
                             (ranked_df['1st metric'] == 'Max Cut Validation Loss') |
                             (ranked_df['1st metric'] == '2nd Cut Validation Loss') |
                             (ranked_df['1st metric'] == '3rd Cut Validation Loss') |
                             (ranked_df['1st metric'] == '4th Cut Validation Loss')), :]

    for this_tech in hpo_techs:

        this_distance_value = np.nanmean(setup_df.loc[(setup_df['1st metric HPO-technique'] == this_tech),
                                                      '1st metric scaled deviation'].to_numpy())

        dv_per_setup_df.loc[idx, 'HPO technique'] = hpo_map_dict[this_tech]
        dv_per_setup_df.loc[idx,
                            'Average Distance Value'] = this_distance_value
        dv_per_setup_df.loc[idx, 'Benchmarking Setup'] = setup_labels[i]

        idx += 1

sns.set_theme(font='Arial', style='ticks')

cat = sns.catplot(data=dv_per_setup_df, x='HPO technique', y='Average Distance Value',
                  hue='Benchmarking Setup', kind='bar', legend_out=False,
                  palette=sns.color_palette(['#179c7d', '#8bcdbe', '#dddddd']))

cat.fig.set_size_inches(6.5, 4)
cat.set(xlabel=None)

plt.tick_params(axis='both', labelsize=11)
plt.tick_params(axis='x', rotation=90)


fig_name = './hpo_framework/results/%s/RankingAnalysis/dv_per_setup_%s.svg' % (
    dataset, dataset)
cat.fig.savefig(fig_name, bbox_inches='tight')
