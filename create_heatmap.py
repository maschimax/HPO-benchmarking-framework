import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'turbofan'
algo_dict = {'AdaBoostRegressor': 'AdaBoost',
             'AdaBoostClassifier': 'AdaBoost',
             'DecisionTreeRegressor': 'Decision Tree',
             'DecisionTreeClassifier': 'Decision Tree',
             'SVR': 'SVM',
             'SVC': 'SVM',
             'KNNRegressor': 'K-NN',
             'KNNClassifier': 'K-NN',
             'LGBMRegressor': 'LightGBM',
             'LGBMClassifier': 'LightGBM',
             'RandomForestRegressor': 'Random Forest',
             'RandomForestClassifier': 'Random Forest',
             'XGBoostRegressor': 'XGBoost',
             'XGBoostClassifier': 'XGBoost',
             'LogisticRegression': 'Logistic Regression',
             'ElasticNet': 'Elastic Net',
             'KerasRegressor': 'MLP',
             'KerasClassifier': 'MLP',
             'NaiveBayes': 'Naive Bayes'}

file_path = './hpo_framework/results/%s/RankingAnalysis/%s_ranked_metrics.csv' % (
    dataset, dataset)

ranked_df = pd.read_csv(file_path, index_col=0)

sub_df = ranked_df.loc[(ranked_df['1st metric'] == 'Max Cut Validation Loss') |
                       (ranked_df['1st metric'] == '2nd Cut Validation Loss') |
                       (ranked_df['1st metric'] == '3rd Cut Validation Loss') |
                       (ranked_df['1st metric'] == '4th Cut Validation Loss'), :]

ml_algorithms = sub_df['ML-algorithm'].unique()
hpo_techs = sub_df['1st metric HPO-technique'].unique()

# heat_arr = np.zeros(shape=(len(ml_algorithms), len(hpo_techs)))

heat_df = pd.DataFrame([])

for this_algo in ml_algorithms:

    for this_tech in hpo_techs:

        avg_scaled_dev = np.nanmean(sub_df.loc[(sub_df['ML-algorithm'] == this_algo) &
                                               (sub_df['1st metric HPO-technique'] == this_tech), '1st metric scaled deviation'].to_numpy())

        heat_df.loc[algo_dict[this_algo], this_tech] = round(avg_scaled_dev, 3)

# Reorder the columns
ordered_heat_df = pd.DataFrame([])
ordered_heat_df[['Default HPs', 'Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB', 'FABOLAS', 'BOHAMIANN']
                ] = heat_df[['Default HPs', 'RandomSearch', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB', 'Fabolas', 'Bohamiann']]

# Sort index (ML algorithms) in alphabetical order
ordered_heat_df.sort_index(axis=0, ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
font_size = 11
sns.set_theme(font='Arial')

cmap = sns.light_palette(color='#179c7d', n_colors=10,
                         as_cmap=True, reverse=True)

sns.heatmap(data=ordered_heat_df, annot=True, linewidths=0.5, cbar=True, square=True,
            ax=ax, vmax=1.0, vmin=0.0, cmap=cmap, annot_kws={'fontsize': 9, 'color': 'black'})

ax.tick_params(axis='both', labelsize=11)

fig_name = './hpo_framework/results/%s/RankingAnalysis/heatmap_%s.svg' % (
    dataset, dataset)
plt.savefig(fig_name, bbox_inches='tight')
