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
# HP space for fully connected neural network from: https://arxiv.org/pdf/1905.04970.pdf
space_keras = [skopt.space.Categorical([.0005, .001, .005, .01, .1], name='init_lr'),
               skopt.space.Categorical([8, 16, 32, 64], name='batch_size'),
               skopt.space.Categorical(['cosine', 'constant'], name='lr_schedule'),
               skopt.space.Categorical(['relu', 'tanh'], name='layer1_activation'),
               skopt.space.Categorical(['relu', 'tanh'], name='layer2_activation'),
               skopt.space.Categorical([16, 32, 64, 128, 256, 512], name='layer1_size'),
               skopt.space.Categorical([16, 32, 64, 128, 256, 512], name='layer2_size'),
               skopt.space.Categorical([.0, .3, .6], name='dropout1'),
               skopt.space.Categorical([.0, .3, .6], name='dropout2')]

# HP values for warm starting a Keras model //
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
warmstart_keras = {'init_lr': 0.001,
                   'batch_size': 200,
                   'lr_schedule': 'constant',
                   'layer1_activation': 'relu',
                   'layer2_activation': 'relu',
                   'layer1_size': 100,
                   'layer2_size': 100,
                   'dropout1': 0.0,
                   'dropout2': 0.0}

# XGBoostModel
# HP space used by Hanno
space_xgb = [skopt.space.Categorical(['gbtree', 'gblinear', 'dart'], name='booster'),
             skopt.space.Integer(1, 200, name='n_estimators'),
             skopt.space.Real(0.0, 1, name='eta'),
             skopt.space.Integer(1, 80, name='max_depth'),
             skopt.space.Categorical(['uniform', 'weighted'], name='sample_type'),
             skopt.space.Categorical(['tree', 'forest'], name='normalize_type'),
             skopt.space.Real(0.0, 1.0, name='rate_drop'),
             skopt.space.Categorical(['shotgun', 'coord_descent'], name='updater')]

# Default / warm start parameters for an XGBoostModel: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
warmstart_xgb = {'booster': 'gbtree',
                 'n_estimators': 100,
                 'eta': 0.3,
                 'max_depth': 6,
                 'sample_type': 'uniform',
                 'normalize_type': 'tree',
                 'rate_drop': 0.0,
                 'updater': 'shotgun'}

# AdaBoostRegressor
# First try >>> iterative testing to find meaningful ranges for each HP // or refer to the AdaBoost literature
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

# LightGBM model
# Important HPs: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# Example of HP tuning with Optuna:
# https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
space_lgb = [skopt.space.Integer(2, 256, name='num_leaves'),
             skopt.space.Integer(20, 1000, name='min_data_in_leaf'),
             skopt.space.Integer(-1, 100, name='max_depth'),
             skopt.space.Real(low=0.0, high=10.0, name='lambda_l1'),
             skopt.space.Real(low=0.0, high=10.0, name='lambda_l2')]

# HP values for warm starting a LightGBM model // HP values need to be inside the bounds of the predefined HP-space
# for a LightGBM model (see above)
warmstart_lgb = {'num_leaves': 31,
                 'min_data_in_leaf': 20,
                 'max_depth': -1,
                 'lambda_l1': 0.0,
                 'lambda_l2': 0.0}

# Logistic Regression
# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
# penalty, tol, C, fit_intercept, intercept_scaling, solver, max_iter
space_logr = [skopt.space.Real(low=1e-6, high=1e-2, name='tol'),
              skopt.space.Real(low=0.5, high=2.0, name='C'),
              skopt.space.Categorical([True, False], name='fit_intercept'),
              skopt.space.Real(low=0.5, high=2.0, name='intercept_scaling'),
              skopt.space.Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], name='solver'),
              skopt.space.Integer(low=50, high=150, name='max_iter')]

# skopt.space.Categorical(['l1', 'l2', 'elasticnet', 'none'], name='penalty'),

# Gaussian Naive Bayes
space_nb = [skopt.space.Real(low=1e-12, high=1e-2, name='var_smoothing')]
