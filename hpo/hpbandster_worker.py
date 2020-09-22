from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import skopt


class HPBandsterWorker(Worker):
    def __init__(self, x_train, x_val, y_train, y_val, ml_algorithm, optimizer_object, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)

        self.X_train = x_train
        self.X_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.ml_algorithm = ml_algorithm
        self.optimizer_object = optimizer_object
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, *args, **kwargs):  # <<< ersetzt die objective function aus der optimizer Klasse?

        # >>> Varying random seed <<<
        # >>> Budget <<<

        # Select the corresponding objective function of the ML-Algorithm
        if self.ml_algorithm == 'RandomForestRegressor' or self.ml_algorithm == 'SVR':
            eval_func = self.optimizer_object.train_evaluate_scikit_regressor

        # elif self.ml_algorithm == 'KerasRegressor':
        #     eval_func = self.train_evaluate_keras_regressor

        # elif self.ml_algorithm == 'XGBoostRegressor':
        #     eval_func = self.train_evaluate_xgboost_regressor

        else:
            raise NameError('Unknown ML-algorithm!')

        val_loss = eval_func(params=config)

        return ({'loss': val_loss,
                 'info': {'validation_loss': val_loss}})

    @staticmethod
    def get_configspace(hp_space: dict):
        cs = CS.ConfigurationSpace()

        # hyperparameters
        params_list = []
        for i in range(len(hp_space)):

            if type(hp_space[i]) == skopt.space.space.Integer:
                params_list.append(CSH.UniformIntegerHyperparameter(name=hp_space[i].name,
                                                                    lower=hp_space[i].low,
                                                                    upper=hp_space[i].high))

            elif type(hp_space[i]) == skopt.space.space.Categorical:
                params_list.append(CSH.CategoricalHyperparameter(hp_space[i].name,
                                                                 choices=list(hp_space[i].categories)))

            elif type(hp_space[i]) == skopt.space.space.Real:
                params_list.append(CSH.UniformFloatHyperparameter(hp_space[i].name,
                                                                  lower=hp_space[i].low,
                                                                  upper=hp_space[i].high))

            else:
                raise NameError('The skopt HP-space could not be converted correctly!')

        cs.add_hyperparameters(params_list)

        return cs