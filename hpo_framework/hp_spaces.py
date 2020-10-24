import skopt

# Hyperparameter spaces according to skopt
# Random Forest Regressor
space_rf_reg = [skopt.space.Integer(1, 200, name='n_estimators'),
                skopt.space.Integer(1, 80, name='max_depth'),
                skopt.space.Integer(1, 30, name='min_samples_leaf'),
                skopt.space.Integer(2, 20, name='min_samples_split'),
                skopt.space.Categorical(['auto', 'sqrt'], name='max_features')]

# Random Forest Classifier
space_rf_clf = [skopt.space.Integer(1, 80, name='max_depth'),
                skopt.space.Integer(1, 30, name='min_samples_leaf'),
                skopt.space.Integer(2, 20, name='min_samples_split'),
                skopt.space.Categorical(['auto', 'sqrt'], name='max_features'),
                skopt.space.Integer(10, 200, name='n_estimators'),
                skopt.space.Categorical(['balanced', None], name='class_weight')]

# SVM-Regressor
space_svr = [skopt.space.Real(low=1e-3, high=1e+3, name='C'),
             skopt.space.Categorical(['scale', 'auto'], name='gamma'),
             skopt.space.Real(low=1e-3, high=1e+0, name='epsilon')]

# SVM-Classifier
space_svc = [skopt.space.Real(low=1e-3, high=1e+3, name='C'),
             skopt.space.Real(low=1e-3, high=1e+3, name='gamma'),
             skopt.space.Categorical(['sigmoid', 'rbf', 'poly'], name='kernel'),
             skopt.space.Categorical(['balanced', None], name='class_weight')]

# KerasRegressor
space_keras = [skopt.space.Categorical([.0005, .001, .005, .01, .1], name='init_lr'),
               skopt.space.Categorical([8, 16, 32, 64], name='batch_size'),
               skopt.space.Categorical(['cosine', 'constant'], name='lr_schedule'),
               skopt.space.Categorical(['relu', 'tanh'], name='layer1_activation'),
               skopt.space.Categorical(['relu', 'tanh'], name='layer2_activation'),
               skopt.space.Categorical([16, 32, 64, 128, 256, 512], name='layer1_size'),
               skopt.space.Categorical([16, 32, 64, 128, 256, 512], name='layer2_size'),
               skopt.space.Categorical([.0, .3, .6], name='dropout1'),
               skopt.space.Categorical([.0, .3, .6], name='dropout2')]

# XGBoostRegressor
space_xgb = [skopt.space.Categorical(['gbtree', 'gblinear', 'dart'], name='booster'),
             skopt.space.Integer(1, 200, name='n_estimators'),
             skopt.space.Integer(1, 80, name='max_depth')]

# AdaBoostRegressor
# First try >>> iterative testing to find meaningful ranges for each HP // or refer the AdaBoost literature
space_ada = [skopt.space.Integer(1, 200, name='n_estimators'),
             skopt.space.Real(.1, 5, name='learning_rate'),
             skopt.space.Categorical(['linear', 'square', 'exponential'], name='loss')]

# DecisionTreeRegressor
# Hanno treated all integer-HPs as a continuous HP >> only continuous HPs (CMA-ES is applicable)
# A lot more HPs in scikit-learn documentation (e.g. max_features; included in RF HP-space)
space_dt = [skopt.space.Integer(2, 20, name='min_samples_split'),
            skopt.space.Integer(1, 80, name='max_depth'),
            skopt.space.Integer(1, 30, name='min_samples_leaf')]

# Linear Regression
space_linr = [skopt.space.Categorical([True, False], name='fit_intercept'),
              skopt.space.Categorical([False, True], name='normalize')]

# KNNRegressor (KNeighborsRegressor)
space_knn_reg = [skopt.space.Integer(1, 10, name='n_neighbors'),
                 skopt.space.Categorical(['uniform', 'distance'], name='weights'),
                 skopt.space.Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
                 skopt.space.Integer(1, 60, name='leaf_size'),
                 skopt.space.Integer(1, 2, name='p')]

# LightGBM Classifier
# Important HPs: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# Example of HP tuning with Optuna:
# https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
space_lgb_clf = [skopt.space.Integer(2, 256, name='num_leaves'),
                 skopt.space.Integer(20, 1000, name='min_data_in_leaf'),
                 skopt.space.Integer(1, 100, name='max_depth'),
                 skopt.space.Real(low=1e-8, high=10.0, name='lambda_l1'),
                 skopt.space.Real(low=1e-8, high=10.0, name='lambda_l2')]