import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

dataset = 'surface'
os.chdir('..')

file_path = './hpo_framework/results/%s/RankingAnalysis/%s_ranked_metrics.csv' % (
    dataset, dataset)

ranked_df = pd.read_csv(file_path, index_col=0)

hpo_techs = ['Default HPs', 'RandomSearch', 'GPBO', 'SMAC', 'TPE', 'Hyperband',
             'BOHB', 'Fabolas', 'Bohamiann']

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

# IPT-colors
ipt_colors = {'Default HPs': '#969696',  # grau
              'BOHB': '#179c7d',  # dunkelgrün
              'CMA-ES': '#ff6600',  # dunkelorange
              'RandomSearch': '#771c2d',  # bordeaux
              'Hyperband': '#438cd4',  # hellblau
              'TPE': '#005a94',  # dunkelblau
              'GPBO': '#b1c800',  # kaktus
              'SMAC': '#25b9e2',  # kaugummi
              'Fabolas': '#ffcc99',  # hellorange
              'Bohamiann': '#000000'}  # schwarz

tb_cats = [(0, 50), (50, 100), (100, 250), (250, 500),
           (500, 1000), (1000, 2000), (2000, 100000)]
tb_labels = ['t <= 50s', '50s < t <= 100s', '100s < t <= 250s', '250s < t <= 500s',
             '500s < t <= 1000s', '1000s < t <= 2000s', '2000s < t']

sub_df = ranked_df.loc[(ranked_df['1st metric'] == 'Max Cut Validation Loss') |
                       (ranked_df['1st metric'] == '2nd Cut Validation Loss') |
                       (ranked_df['1st metric'] == '3rd Cut Validation Loss') |
                       (ranked_df['1st metric'] == '4th Cut Validation Loss'), :]

fig, ax = plt.subplots(figsize=(9, 5))

# for this_tech in hpo_techs:

#     avg_dv = []

#     for budget_cat in tb_cats:

#         # No time budgets defined for Default HPs
#         if this_tech == 'Default HPs':

#             this_dv = np.nanmean(sub_df.loc[(sub_df['1st metric HPO-technique'] == this_tech), '1st metric scaled deviation'].to_numpy())

#         else:

#             this_dv = np.nanmean(sub_df.loc[(sub_df['1st metric HPO-technique'] == this_tech) &
#                                             (sub_df['Time budget [s]'] > budget_cat[0]) &
#                                             (sub_df['Time budget [s]'] <= budget_cat[1]), '1st metric scaled deviation'].to_numpy())

#         avg_dv.append(this_dv)

#     # ax.plot(range(len(tb_cats)), avg_dv, label=hpo_map_dict[this_tech])
#     ax = sns.regplot(x=np.arange(0, len(tb_cats), 1), y=avg_dv,
#                      label=hpo_map_dict[this_tech], order=1, color=ipt_colors[this_tech], ci=30)

# ax.legend()
# ax.set_xlim(-0.25, len(tb_cats)-0.75)
# ax.set_ylim(-0.1, 1.1)
# plt.xticks(np.arange(0, len(tb_cats), 1), tb_labels)
# plt.show()

for this_tech in hpo_techs:

    hpo_df = sub_df.loc[(sub_df['1st metric HPO-technique']
                         == this_tech), :].copy(deep=True)

    hpo_df.sort_values(by='Time budget [s]',
                       axis=0, ascending=True, inplace=True)

    time_budgets = hpo_df['Time budget [s]'].to_numpy()
    distance_values = hpo_df['1st metric scaled deviation'].to_numpy()

    ax = sns.regplot(x=time_budgets, y=distance_values, label=hpo_map_dict[this_tech],
                     color=ipt_colors[this_tech], ci=30, order=1)

    # hpo_df.sort_values(by='Time per eval on this setup (RS)',
    #                    axis=0, ascending=True, inplace=True)

    # t_per_eval = hpo_df['Time per eval on this setup (RS)'].to_numpy()
    # distance_values = hpo_df['1st metric scaled deviation'].to_numpy()

    # ax = sns.regplot(x=t_per_eval, y=distance_values, label=hpo_map_dict[this_tech],
    #                  color=ipt_colors[this_tech], ci=30, order=1)

plt.xscale('log')
ax.set_ylim(-0.1, 1.1)
ax.legend()
plt.show()
