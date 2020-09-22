from hpbandster.core.worker import Worker
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class HPBandsterWorker(Worker):
    def __init__(self, x_train, x_val, y_train, y_val, ml_algorithm, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)

        self.X_train = x_train
        self.X_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.ml_algorithm = ml_algorithm
        self.sleep_interval = sleep_interval

    def compute(self, config, budget):  # <<< ersetzt die objective function aus der optimizer Klasse?

        # >>> Varying random seed <<<

        # Select the corresponding ml_algorithm
        if self.ml_algorithm == 'RandomForestRegressor':
            model = RandomForestRegressor(**config)  # <<< use existing train_evaluate_functions here?

        elif self.ml_algorithm == 'SVR':
            model = SVR(**config)

        # elif self.ml_algorithm == 'KerasRegressor':
        #     eval_func = self.train_evaluate_keras_regressor

        # elif self.ml_algorithm == 'XGBoostRegressor':
        #     eval_func = self.train_evaluate_xgboost_regressor

        else:
            raise NameError('Unknown ML-algorithm!')

        # Train the model on the specified budget
        pass

    @staticmethod
    def get_configspace():
        pass