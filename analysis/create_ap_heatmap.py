import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('..')

# Create heatmap on dataset or benchmarking study level
aggregation_level = 'benchmarking'  # 'benchmarking', 'dataset'

datasets = ['turbofan', 'scania', 'sensor', 'blisk', 'surface']

# Specify the dataset, if the heatmap is created on dataset level
dataset = 'surface'

# Use: ['t outperform default [s]', 'time'] or: ['Max Cut AUC', 'AUC']
ap_analysis_type = ['t outperform default [s]', 'time']

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

if aggregation_level == 'dataset':

    file_path = './hpo_framework/results/%s/RankingAnalysis/%s_ranked_metrics.csv' % (
        dataset, dataset)

    ranked_df = pd.read_csv(file_path, index_col=0)

elif aggregation_level == 'benchmarking':

    i = 0

    for this_dataset in datasets:

        file_path = './hpo_framework/results/%s/RankingAnalysis/%s_ranked_metrics.csv' % (
            this_dataset, this_dataset)

        if i == 0:

            ranked_df = pd.read_csv(file_path, index_col=0)

        else:

            this_df = pd.read_csv(file_path, index_col=0)

            ranked_df = pd.concat(
                objs=[ranked_df, this_df], axis=0, ignore_index=True)

        i += 1

    ranked_df.reset_index(drop=True, inplace=True)

else:

    raise Exception('Unknown aggregation level!')

sub_df = ranked_df.loc[(ranked_df['1st metric'] == ap_analysis_type[0]), :]

sub_df['ML-algorithm'] = sub_df['ML-algorithm'].map(algo_dict)

ml_algorithms = sub_df['ML-algorithm'].unique()
hpo_techs = sub_df['1st metric HPO-technique'].unique()

# heat_arr = np.zeros(shape=(len(ml_algorithms), len(hpo_techs)))

heat_df = pd.DataFrame([])

for this_algo in ml_algorithms:

    for this_tech in hpo_techs:

        dv_ser = sub_df.loc[(sub_df['ML-algorithm'] == this_algo) &
                            (sub_df['1st metric HPO-technique'] == this_tech), '1st metric scaled deviation']

        dv_ser[dv_ser == np.inf] = 1.0

        avg_scaled_dev = np.nanmean(dv_ser.to_numpy())

        heat_df.loc[this_algo, this_tech] = round(avg_scaled_dev, 3)

# Reorder the columns
ordered_heat_df = pd.DataFrame([])
ordered_heat_df[['Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB', 'FABOLAS', 'BOHAMIANN']
                ] = heat_df[['RandomSearch', 'GPBO', 'SMAC', 'TPE', 'CMA-ES', 'Hyperband', 'BOHB', 'Fabolas', 'Bohamiann']]

# Sort index (ML algorithms) in alphabetical order
ordered_heat_df.sort_index(axis=0, ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
font_size = 11
sns.set_theme(font='Arial')

cmap = sns.light_palette(color='#176290', n_colors=10,
                         as_cmap=True, reverse=True)

sns.heatmap(data=ordered_heat_df, annot=True, linewidths=0.5, cbar=True, square=True,
            ax=ax, vmax=1.0, vmin=0.0, cmap=cmap, annot_kws={'fontsize': 9, 'color': 'black'})

ax.tick_params(axis='both', labelsize=11)

if aggregation_level == 'dataset':

    fig_name = './hpo_framework/results/%s/RankingAnalysis/heatmap_%s_%s.svg' % (
        dataset, ap_analysis_type[1], dataset)

elif aggregation_level == 'benchmarking':

    fig_name = './hpo_framework/results/heatmap_overall_%s.svg' % ap_analysis_type[1]

else:

    raise Exception('Unknown aggregation level!')

plt.savefig(fig_name, bbox_inches='tight')
