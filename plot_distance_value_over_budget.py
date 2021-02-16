import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'turbofan'

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

sub_df = ranked_df.loc[(ranked_df['1st metric'] == 'Max Cut Validation Loss') |
                       (ranked_df['1st metric'] == '2nd Cut Validation Loss') |
                       (ranked_df['1st metric'] == '3rd Cut Validation Loss') |
                       (ranked_df['1st metric'] == '4th Cut Validation Loss'), :]

# fig, ax = plt.subplots(figsize=(9, 5))

for this_tech in hpo_techs:

    hpo_df = sub_df.loc[(sub_df['1st metric HPO-technique'] == this_tech), :].copy(deep=True)

    hpo_df.sort_values(by='Time budget [s]', axis=0, ascending=True, inplace=True)

    time_budgets = hpo_df['Time budget [s]'].to_numpy()
    distance_values = hpo_df['1st metric scaled deviation'].to_numpy()

    # ax.scatter(time_budgets, distance_values, label=hpo_map_dict[this_tech])
    ax = sns.regplot(x=time_budgets, y=distance_values, order=3)

# ax.legend()
plt.xscale('log')
ax.set_ylim(-0.1, 1.1)
plt.show()
