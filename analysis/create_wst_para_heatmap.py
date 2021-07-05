import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

influence_factor = 'warm-start'  # 'parallelization' or 'warm-start'
metric = 'FP'  # 'FP' or 'AP'

# Mappings
metrics_dict = {'FP': 'Mean (final validation loss)',
                'AP': 't outperform default [s]'}

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

# Create heatmap data
heat_df = pd.DataFrame([])

# Iterate over data sets
datasets = ['turbofan', 'scania', 'sensor', 'blisk', 'surface']

idx = 0
os.chdir('..')

for this_dataset in datasets:

    file_path = './hpo_framework/results/%s/metrics_%s.csv' % (this_dataset,
                                                               this_dataset)

    metrics_df = pd.read_csv(file_path, index_col=0)

    # Map ML-algorithms
    metrics_df['ML-algorithm'] = metrics_df['ML-algorithm'].map(algo_dict)

    ml_algos = metrics_df['ML-algorithm'].unique()
    hpo_techs = metrics_df['HPO-method'].unique()

    for this_algo in ml_algos:

        for this_tech in hpo_techs:

            this_df = metrics_df.loc[(metrics_df['ML-algorithm'] == this_algo) &
                                     (metrics_df['HPO-method'] == this_tech), :]

            if influence_factor == 'parallelization':

                basic_value = this_df.loc[(this_df['Workers'] == 1) &
                                          (this_df['Warmstart'] == False), metrics_dict[metric]].values

                affected_value = this_df.loc[(this_df['Workers'] == 8) &
                                             (this_df['Warmstart'] == False), metrics_dict[metric]].values

            elif influence_factor == 'warm-start':

                basic_value = this_df.loc[(this_df['Workers'] == 1) &
                                          (this_df['Warmstart'] == False), metrics_dict[metric]].values

                affected_value = this_df.loc[(this_df['Workers'] == 1) &
                                             (this_df['Warmstart'] == True), metrics_dict[metric]].values

            else:

                raise Exception('Unknown setup variable!')

            if len(affected_value) == 0 or len(basic_value) == 0 or affected_value[0] == np.inf or basic_value[0] == np.inf:

                continue

            rel_change = (affected_value - basic_value) / basic_value

            heat_df.loc[idx, 'ML-algorithm'] = this_algo
            heat_df.loc[idx, 'HPO-technique'] = this_tech
            heat_df.loc[idx, 'Relative change'] = rel_change
            heat_df.loc[idx, 'Dataset'] = this_dataset

            idx += 1

# Average the relative changes for each pair of ML algorithms and HPO techniques
avg_heat_df = pd.DataFrame([])

for this_algo in heat_df['ML-algorithm'].unique():

    for this_tech in heat_df['HPO-technique'].unique():

        avg_change = np.nanmean(heat_df.loc[(heat_df['ML-algorithm'] == this_algo) &
                                            (heat_df['HPO-technique'] == this_tech), 'Relative change'].to_numpy())
        avg_heat_df.loc[this_algo, this_tech] = round(avg_change, 4)

# Reorder the columns
ordered_heat_df = pd.DataFrame([])

# TODO: HOW TO DEAL WITH BOHAMIANN
if influence_factor == 'parallelization':

    ordered_heat_df[['Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB']
                    ] = avg_heat_df[['RandomSearch', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB']]

elif influence_factor == 'warm-start':

    ordered_heat_df[['Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB', 'BOHAMIANN']
                    ] = avg_heat_df[['RandomSearch', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB', 'Bohamiann']]

else:

    raise Exception('Unknown setup variable!')

# Sort index (ML algorithms) in alphabetical order
ordered_heat_df.sort_index(axis=0, ascending=True, inplace=True)

# Create heatmap plot
fig, ax = plt.subplots(figsize=(9, 5))
font_size = 11
sns.set_theme(font='Arial')

cmap = sns.diverging_palette(h_neg=166, h_pos=20, n=20, center='light',
                             as_cmap=True)

sns.heatmap(data=ordered_heat_df, annot=True, linewidths=0.5, cbar=True, square=True,
            ax=ax, vmax=0.2, vmin=-0.2, cmap=cmap, annot_kws={'fontsize': 9, 'color': 'black'})

ax.tick_params(axis='both', labelsize=11)

fig_name = './hpo_framework/results/heatmap_%s_%s.svg' % (influence_factor, metric)

plt.savefig(fig_name, bbox_inches='tight')
