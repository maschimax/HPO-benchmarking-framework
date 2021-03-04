import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'sensor'
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

file_path = './hpo_framework/results/%s/metrics_%s.csv' % (dataset, dataset)

metrics_df = pd.read_csv(file_path, index_col=0)

ml_algorithms = metrics_df['ML-algorithm'].unique()
hpo_techs = metrics_df['HPO-method'].unique()

crash_df = pd.DataFrame([])

for this_algo in ml_algorithms:

    for this_tech in hpo_techs:

        n_runs = np.sum(metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) &
                                       (metrics_df['HPO-method'] == this_tech), 'Runs'].to_numpy())
        n_crashes = np.sum(metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) &
                                       (metrics_df['HPO-method'] == this_tech), 'Crashes'].to_numpy())
        crash_ratio = round(n_crashes / n_runs, 2)

        crash_df.loc[algo_dict[this_algo], this_tech] = crash_ratio

# Reorder the columns
ordered_crash_df = pd.DataFrame([])
ordered_crash_df[['Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB', 'FABOLAS', 'BOHAMIANN']
                ] = crash_df[['RandomSearch', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB', 'Fabolas', 'Bohamiann']]

# Sort index (ML algorithms) in alphabetical order
ordered_crash_df.sort_index(axis=0, ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
font_size = 11
sns.set_theme(font='Arial')

cmap = sns.light_palette(color='#268A97', n_colors=10,
                         as_cmap=True, reverse=False)

sns.heatmap(data=ordered_crash_df, annot=True, linewidths=0.5, cbar=True, square=True,
            ax=ax, vmax=1.0, vmin=0.0, cmap=cmap, annot_kws={'fontsize': 9, 'color': 'black'})

ax.tick_params(axis='both', labelsize=11)

fig_name = './hpo_framework/results/%s/crashmap_%s.svg' % (
    dataset, dataset)
plt.savefig(fig_name, bbox_inches='tight')
