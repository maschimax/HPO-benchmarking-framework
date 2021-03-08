import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# IPT-colors
ipt_colors = {
    'Bohamiann': '#000000',
    'BOHB': '#179c7d',
    'CMA-ES': '#ff6600',
    'Fabolas': '#ffcc99',
    'GPBO': '#b1c800',
    'Hyperband': '#438cd4',
    'RandomSearch': '#771c2d',
    'SMAC': '#25b9e2',
    'TPE': '#005a94',
    'Default Values': '#969696',
}

# Map HPO techniques: BRB -> BM
hpo_brb2bm_map = {
    'BOHAMIANN': 'Bohamiann',
    'BOHB': 'BOHB',
    'CMA-ES': 'CMA-ES',
    'FABOLAS': 'Fabolas',
    'GPBO': 'GPBO',
    'HB': 'Hyperband',
    'Random Search': 'RandomSearch',
    'SMAC': 'SMAC',
    'TPE': 'TPE',
    'Default Values': 'Default Values'
}

# Map HPO techniques: BM -> BRB
hpo_bm2brb_map = {v: k for k, v in hpo_brb2bm_map.items()}

# Map warm tart notation: BRB -> BM
wst_brb2bm_map = {'yes': True, 'no': False}

wst_bm2brb_map = {v: k for k, v in wst_brb2bm_map.items()}

# Map ML algorithms with HP data types
bmalgo2paratype_map = {
    'RandomForestRegressor': '[continuous, discrete, nominal]',
    'RandomForestClassifier': '[continuous, discrete, nominal]',
    'MLPRegressor': '[discrete, nominal]',
    'MLPClassifier': '[discrete, nominal]',
    'SVR': '[continuous, nominal]',
    'SVC': '[continuous, nominal]',
    'KerasRegressor': '[continuous, discrete, nominal]',
    'KerasClassifier': '[continuous, discrete, nominal]',
    'XGBoostClassifier': '[continuous, discrete, nominal]',
    'XGBoostRegressor': '[continuous, discrete, nominal]',
    'AdaBoostRegressor': '[continuous, discrete, nominal]',
    'AdaBoostClassifier': '[continuous, discrete, nominal]',
    'DecisionTreeRegressor': '[continuous, discrete]',
    'DecisionTreeClassifier': '[continuous, discrete]',
    'LinearRegression': '[nominal]',
    'KNNRegressor': '[discrete, nominal]',
    'KNNClassifier': '[discrete, nominal]',
    'LGBMRegressor': '[continuous, discrete]',
    'LGBMClassifier': '[continuous, discrete]',
    'LogisticRegression': '[continuous, discrete, nominal]',
    'ElasticNet': '[continuous, discrete, nominal]',
    'NaiveBayes': '[continuous]'}

# Map ML algorithms with conditional / non-conditional HPs
bmalgo2cond_map = {
    'RandomForestRegressor': 'no',
    'RandomForestClassifier': 'no',
    'MLPRegressor': 'no',
    'MLPClassifier': 'no',
    'SVR': 'no',
    'SVC': 'no',
    'KerasRegressor': 'no',
    'KerasClassifier': 'no',
    'XGBoostClassifier': 'yes',
    'XGBoostRegressor': 'yes',
    'AdaBoostRegressor': 'no',
    'AdaBoostClassifier': 'no',
    'DecisionTreeRegressor': 'no',
    'DecisionTreeClassifier': 'no',
    'LinearRegression': 'no',
    'KNNRegressor': 'no',
    'KNNClassifier': 'no',
    'LGBMRegressor': 'no',
    'LGBMClassifier': 'no',
    'LogisticRegression': 'no',
    'ElasticNet': 'no',
    'NaiveBayes': 'no'}

gpu_cpu_map = {
    False: 'CPU',
    True: 'GPU'
}

# Map dataset information with some (constant) antecedents
datset2constant_map = {
    'turbofan': {
        "Detailed ML task": 'Remaining Useful Lifetime',
        "Production application area": 'Predictive Maintenance',
        "Input Data": 'Tabular Data',
        "Ratio training to test dataset": 4,
        "ML task": 'Regression',
        "Loss function": 'customized',
        "Special properties of loss function": 'Exponential'
    },
    'scania': {
        "Detailed ML task": 'Part Failure',
        "Production application area": 'Predictive Maintenance',
        "Input Data": 'Tabular Data',
        "Ratio training to test dataset": 4,
        "ML task": 'Binary Classification',
        "Loss function": 'F1-loss',
        "Special properties of loss function": ''
    },
    'sensor': {
        "Detailed ML task": 'Product Quality',
        "Production application area": 'Predictive Quality',
        "Input Data": 'Tabular Data',
        "Ratio training to test dataset": 4,
        "ML task": 'Multiclass Classification',
        "Loss function": 'F1-loss',
        "Special properties of loss function": ''
    },
    'blisk': {
        "Detailed ML task": 'Time Series Prediction',
        "Production application area": 'Process Parameter Prediction',
        "Input Data": 'Tabular Data',
        "Ratio training to test dataset": 4,
        "ML task": 'Regression',
        "Loss function": 'RMSE',
        "Special properties of loss function": ''
    },
    'surface': {
        "Detailed ML task": 'Image Recognition',
        "Production application area": 'Predictive Quality',
        "Input Data": 'Tabular Data',
        "Ratio training to test dataset": 4,
        "ML task": 'Binary Classification',
        "Loss function": 'F1-loss',
        "Special properties of loss function": ''
    }
}

hpo_techs = ['Bohamiann', 'BOHB', 'CMA-ES', 'Fabolas', 'GPBO',
             'Hyperband', 'RandomSearch', 'SMAC', 'TPE', 'Default Values']

# Drop constant, unused and empty columns
drop_cols = ['Obtainability of gradients',
             'Obtainability of good approximate',
             'Supports parallel evaluations',
             'Number of maximum function evaluations/ trials budget']


def preprocess_X(X_data, validation_mode):

    y_data = X_data.loc[:, hpo_techs].copy(deep=True)
    X_data.drop(hpo_techs, axis=1, inplace=True)

    X_data.drop(drop_cols, axis=1, inplace=True)

    # Handle remaining NaN values
    X_data.loc[:, 'Special properties of loss function'].fillna(
        value='no', inplace=True)
    X_data.loc[:, 'Running time per trial [s]'].fillna(
        value=np.nanmean(X_data['Running time per trial [s]']), inplace=True)

    # Impute label NaNs (HPO technique not evaluated in this use case)
    for this_hpo in y_data.columns:
        y_data.loc[:, this_hpo].fillna(value=1.0, inplace=True)

    # Encoding of HP types
    hp_types = ['continuous', 'discrete', 'nominal']
    for idx, row in X_data.iterrows():

        for this_type in hp_types:

            if this_type in row['HP datatypes']:
                X_data.loc[idx, this_type] = 1
            else:
                X_data.loc[idx, this_type] = 0

    X_data.drop('HP datatypes', axis=1, inplace=True)

    # One-Hot-Encoding of categorical features
    cat_cols = [col for col in X_data.columns if X_data[col].dtype == 'object']
    X_cat_oh = pd.get_dummies(X_data[cat_cols])

    # Scaling of numerical features
    num_cols = set(X_data.columns) - set(cat_cols)

    if validation_mode in ['FI-overall', 'FI-dataset']:

        scaler = StandardScaler()
        X_num = pd.DataFrame(scaler.fit_transform(
            X_data[num_cols]), columns=num_cols)

    elif validation_mode == 'hold-out':

        X_num = X_data[num_cols].copy(deep=True)

    else:
        raise Exception('Unknown validation mode!')

    # Concatenate numerical and categorical features
    X_processed = pd.concat(objs=[X_num, X_cat_oh], axis=1)

    return X_processed, y_data


if __name__ == '__main__':

    metrics_folder = 'C:/Users/Max/Desktop/BM_results'
    create_X_data = False
    validation_set = 'blisk'
    validation_mode = 'FI-dataset'  # 'FI-dataset', 'FI-overall', 'hold-out'

    # Extract use cases and labels from the metrics files (create training data set)
    if create_X_data:

        datasets = {'turbofan': 'Regression',
                    'scania': 'Binary Classification',
                    'sensor': 'Multiclass Classification',
                    'blisk': 'Regression',
                    'surface': 'Binary Classification'}

        ur_penalty = 0.05

        dataset_idx = 0

        for this_dataset in datasets.keys():

            file_name = 'expanded_metrics_' + this_dataset + '.csv'
            metric_df = pd.read_csv(os.path.join(
                metrics_folder, file_name), index_col=0)

            # Mapping

            ml_task = datasets[this_dataset]

            if ml_task == 'Regression':

                ml_brb2bm_map = {
                    'AdaBoost': 'AdaBoostRegressor',
                    'Decision Tree': 'DecisionTreeRegressor',
                    'Support Vector Machine': 'SVR',
                    'KNN': 'KNNRegressor',
                    'LightGBM': 'LGBMRegressor',
                    'Random Forest': 'RandomForestRegressor',
                    'XGBoost': 'XGBoostRegressor',
                    'Elastic Net': 'ElasticNet',
                    'Multilayer Perceptron': 'KerasRegressor'
                }

            elif ml_task == 'Binary Classification' or ml_task == 'Multiclass Classification':

                ml_brb2bm_map = {
                    'AdaBoost': 'AdaBoostClassifier',
                    'Decision Tree': 'DecisionTreeClassifier',
                    'Support Vector Machine': 'SVC',
                    'KNN': 'KNNClassifier',
                    'LightGBM': 'LGBMClassifier',
                    'Random Forest': 'RandomForestClassifier',
                    'XGBoost': 'XGBoostClassifier',
                    'Logistic Regression': 'LogisticRegression',
                    'Multilayer Perceptron': 'KerasClassifier',
                    'Naive Bayes': 'NaiveBayes'
                }

            else:
                raise Exception('Unknown ML task!')

            # Map ML algorithms: BM -> BRB
            ml_bm2brb_map = {v: k for k, v in ml_brb2bm_map.items()}

            # Extract the HPO use cases from the metrics files (serve as input for the ML model)
            df_use_case = pd.DataFrame([])
            df_use_case['ID'] = metric_df['ID']
            df_use_case['Machine Learning Algorithm'] = metric_df['ML-algorithm'].map(
                ml_bm2brb_map)
            df_use_case['Hardware: Number of workers/kernels for parallel computing'] = metric_df['Workers']
            df_use_case['Availability of a warm-start HP configuration'] = metric_df['Warmstart'].map(
                wst_bm2brb_map)
            df_use_case['Number of maximum function evaluations/ trials budget'] = metric_df['Evaluations']
            df_use_case['Running time per trial [s]'] = metric_df['Wall clock time [s]'] / \
                metric_df['Evaluations']
            df_use_case['Total Computing Time [s]'] = metric_df['Wall clock time [s]']
            df_use_case['Dimensionality of HPs'] = metric_df['# cont. HPs'] + \
                metric_df['# int. HPs'] + metric_df['# cat. HPs']
            df_use_case['HP datatypes'] = metric_df['ML-algorithm'].map(
                bmalgo2paratype_map)
            df_use_case['Conditional HP space'] = metric_df['ML-algorithm'].map(
                bmalgo2cond_map)
            df_use_case["Detailed ML task"] = datset2constant_map[this_dataset]["Detailed ML task"]
            df_use_case["Production application area"] = datset2constant_map[this_dataset]["Production application area"]
            df_use_case['Input Data'] = datset2constant_map[this_dataset]["Input Data"]
            df_use_case['#Instances training dataset'] = metric_df['# training instances']
            df_use_case['Ratio training to test dataset'] = datset2constant_map[this_dataset]["Ratio training to test dataset"]
            df_use_case['ML task'] = datset2constant_map[this_dataset]["ML task"]
            df_use_case["UR: need for model transparency"] = metric_df["UR: need for model transparency"]
            df_use_case["UR: Availability of a well documented library"] = metric_df["UR: Availability of a well documented library"]
            df_use_case["User's programming ability"] = metric_df["User's programming ability"]

            # fixed antecedents (cannot yet be derived from the metrics .csv file)
            df_use_case["UR: quality demands"] = metric_df["UR: quality demands"]
            df_use_case["UR: Computer operating system"] = 'Linux'
            df_use_case["Obtainability of good approximate"] = ''
            df_use_case["Supports parallel evaluations"] = ''
            df_use_case["Obtainability of gradients"] = ''
            df_use_case["Noise in dataset"] = 'no'
            df_use_case["Training Technique"] = "Offline"

            df_use_case["CPU / GPU"] = metric_df['GPU'].map(gpu_cpu_map)
            df_use_case["Loss function"] = datset2constant_map[this_dataset]['Loss function']
            df_use_case["Special properties of loss function"] = datset2constant_map[this_dataset]['Special properties of loss function']
            df_use_case['UR: Anytime Performance'] = 'low'
            df_use_case['UR: Robustness'] = metric_df['Robustness']

            # Remove duplicate use cases
            df_use_case.drop_duplicates(inplace=True, ignore_index=True)

            # Create a label for each use case
            for idx, use_case in df_use_case.iterrows():

                exp = metric_df.loc[(metric_df['ML-algorithm'] == ml_brb2bm_map[use_case['Machine Learning Algorithm']]) &
                                    (metric_df['Workers'] == use_case['Hardware: Number of workers/kernels for parallel computing']) &
                                    (metric_df['Warmstart'] == wst_brb2bm_map[use_case['Availability of a warm-start HP configuration']]) &
                                    (metric_df['Wall clock time [s]'] == use_case['Total Computing Time [s]']) &
                                    (metric_df["User's programming ability"] == use_case["User's programming ability"]) &
                                    (metric_df['UR: need for model transparency'] == use_case['UR: need for model transparency']) &
                                    (metric_df['UR: Availability of a well documented library'] == use_case['UR: Availability of a well documented library']) &
                                    (metric_df['UR: quality demands'] == use_case["UR: quality demands"]) &
                                    (metric_df['Robustness'] == use_case['UR: Robustness']), :]
                # TODO: Iterate over Robustness (& Anytime Performance?)

                if len(exp) == 0:
                    raise Exception(
                        'Label creation failed! None of the experiments is matching the requirements!')

                hpos = exp.set_index(
                    'HPO-method')['Mean (final validation loss)']
                hpos.loc['Default Values'] = np.nanmean(
                    exp['Validation baseline'])

                # Compute the scaled loss deviation for each suitable HPO technique in this experiment
                loss_arr = hpos.to_numpy()
                min_value = np.nanmin(loss_arr)
                max_value = np.nanmax(loss_arr[loss_arr != np.inf])
                scaled_hpos = (hpos - min_value) / (max_value - min_value)

                # Filter HPO techniques, that have been classified as non suitable for this use case based on the user requirements and the robustness antecedent
                non_suitable_hpos = exp.loc[exp['HPO suitability']
                                            == 'no', 'HPO-method'].values

                for this_hpo in non_suitable_hpos:
                    scaled_hpos[this_hpo] += ur_penalty

                for this_hpo in scaled_hpos.keys():
                    df_use_case.loc[idx, this_hpo] = scaled_hpos[this_hpo]

            if dataset_idx == 0:

                X_data = df_use_case.copy(deep=True)

            else:

                X_data = pd.concat(
                    objs=[X_data, df_use_case], ignore_index=True, axis=0)

            dataset_idx += 1

            this_file = this_dataset + '_use_cases.csv'
            df_use_case.to_csv(os.path.join(metrics_folder, this_file))

        X_data.to_csv(os.path.join(metrics_folder, 'X_data.csv'))

    # Load the training data

    if validation_mode in ['FI-overall', 'hold-out']:

        X_data = pd.read_csv(os.path.join(
            metrics_folder, 'X_data.csv'), index_col=0)

    elif validation_mode == 'FI-dataset':
        file_name = '%s_use_cases.csv' % validation_set
        X_data = pd.read_csv(os.path.join(
            metrics_folder, file_name), index_col=0)

    else:

        raise Exception('Unknown validation mode!')

    # DATA PREPROCESSING
    X_data.drop('ID', axis=1, inplace=True)
    X_train, y_train = preprocess_X(X_data, validation_mode=validation_mode)
    X_train = X_train.reindex(sorted(X_train.columns), axis=1)

    if validation_mode == 'hold-out':

        test_uc = [('AdaBoost', 1, 'no'), ('AdaBoost',
                                           1, 'yes'), ('AdaBoost', 8, 'no')]

        cv_scores = []

        for this_uc in test_uc:

            # TODO: Shuffle data

            test_idx = X_train.loc[(X_train['Machine Learning Algorithm_' + this_uc[0]] == 1) &
                                   (X_train['Hardware: Number of workers/kernels for parallel computing'] == this_uc[1]) &
                                   (X_train['Availability of a warm-start HP configuration_' + this_uc[2]] == 1), :].index

            X_test_cv = X_train.iloc[test_idx, :]
            y_test_cv = y_train.iloc[test_idx, :]

            X_train_cv = X_train.drop(index=test_idx, inplace=False)
            y_train_cv = y_train.drop(index=test_idx, inplace=False)

            model = RandomForestRegressor(random_state=0)

            model.fit(X_train_cv, y_train_cv)

            y_pred = model.predict(X_test_cv)
            y_pred = pd.DataFrame(
                y_pred, columns=hpo_techs, index=y_test_cv.index)

            scores = []

            for idx, row in y_test_cv.iterrows():

                hpo_rec = y_pred.loc[idx, :].idxmin(axis='columns')
                scores.append(row[hpo_rec])

            cv_scores.append(np.nanmean(scores))

        test_score = np.nanmean(cv_scores)
        print('CV score: ', test_score)
        exit(0)

    elif validation_mode in ['FI-dataset', 'FI-overall']:

        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        cv_scores = []

        for train_idx, test_idx in kfold.split(X_train):
            X_train_cv = X_train.iloc[train_idx, :]
            X_test_cv = X_train.iloc[test_idx, :]
            y_train_cv = y_train.iloc[train_idx, :]
            y_test_cv = y_train.iloc[test_idx, :]

            model = RandomForestRegressor(random_state=0)

            model.fit(X_train_cv, y_train_cv)

            y_pred = model.predict(X_test_cv)
            y_pred = pd.DataFrame(
                y_pred, columns=hpo_techs, index=y_test_cv.index)

            scores = []

            for idx, row in y_test_cv.iterrows():

                hpo_rec = y_pred.loc[idx, :].idxmin(axis='columns')
                scores.append(row[hpo_rec])

            cv_scores.append(np.nanmean(scores))

        test_score = np.nanmean(cv_scores)
        print('CV score: ', test_score)

    else:

        raise Exception('Unknown validation mode!')

    # FEATURE IMPORTANCE
    feat_imp_dict = {}
    for i in range(len(X_train.columns)):
        feat_imp_dict[X_train.columns[i]] = model.feature_importances_[i]

    original_features = set(X_data.columns) - set(drop_cols)

    original_features.add('HP datatypes')
    hp_types = ['continuous', 'discrete', 'nominal']
    for this_type in hp_types:
        original_features.remove(this_type)

    original_feat_imp_dict = {}

    # Reverse One-Hot-Encoding
    for this_feature in original_features:

        original_feat_imp_dict[this_feature] = 0.0

        for k, v in feat_imp_dict.items():

            if this_feature in k:

                original_feat_imp_dict[this_feature] += v

            elif this_feature == 'HP datatypes' and k in hp_types:

                original_feat_imp_dict['HP datatypes'] += v

    # Update some long or misleading keys
    original_feat_imp_dict['Usage of warm-start HP configuration'] = original_feat_imp_dict.pop('Availability of a warm-start HP configuration')
    original_feat_imp_dict['Number of workers'] = original_feat_imp_dict.pop('Hardware: Number of workers/kernels for parallel computing')
    original_feat_imp_dict['UR: Need for model transparency'] = original_feat_imp_dict.pop('UR: need for model transparency')
    original_feat_imp_dict["UR: User's programming ability"] = original_feat_imp_dict.pop("User's programming ability")
    original_feat_imp_dict['UR: Need for a well documented library'] = original_feat_imp_dict.pop('UR: Availability of a well documented library')

    # Sort by descending importance
    feat_imp = pd.Series(original_feat_imp_dict).sort_values(
        ascending=False, inplace=False)

    # Only consider features with FI higher than 0.0
    feat_imp = feat_imp[feat_imp > 0]

    # Create Feature importance plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x=feat_imp.keys(), height=feat_imp.values, color='#3E927F', width=0.6)
    
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_ylabel('Feature Importance', fontsize=11, fontname='Arial')

    file_name_dict = {'FI-dataset': 'feature_importance_%s.svg' % validation_set,
                      'FI-overall': 'feature_importance_overall.svg',
                      'hold-out': 'feature_importance_holdout.svg'}
    

    plt.savefig(os.path.join(metrics_folder,
                             file_name_dict[validation_mode]),
                bbox_inches='tight')

    # # Pie chart - ML expert recommendation
    # fig_pie_expert, ax_pie_expert = plt.subplots()
    # recs = y_pred.idxmin(axis=1)
    # hist_dict = recs.value_counts()
    # ax_pie_expert.pie(x=hist_dict.values, labels=hist_dict.keys(),
    #                   wedgeprops=dict(width=0.3, edgecolor='w'), autopct='%1.1f%%')
    # ax_pie_expert.set_title('ML expert recommendations')
    # plt.savefig('piechart_ml_expert.jpg')

    # # Pie chart - Best HPO techniques based on BM
    # fig_pie_bm, ax_pie_bm = plt.subplots()
    # recs = y_test_cv.idxmin(axis=1)
    # hist_dict = recs.value_counts()
    # ax_pie_bm.pie(x=hist_dict.values, labels=hist_dict.keys(),
    #               wedgeprops=dict(width=0.3, edgecolor='w'), autopct='%1.1f%%')
    # ax_pie_bm.set_title('BM results')
    # plt.savefig('piechart_bm_results.jpg')
